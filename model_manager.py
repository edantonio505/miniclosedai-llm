#!/usr/bin/env python3
"""Core engine for the miniclosedai-llm web control plane.

Pure logic (no FastAPI imports) so it can be unit-tested and reused. Owns:

  * the persistent model registry (`models.local.json`)
  * HuggingFace-id normalization + served-name/port allocation
  * a pluggable launch Engine — `DockerEngine` (normal Ubuntu GPU host) and
    `NativeEngine` (`vllm serve` subprocess, for RunPod pods with no Docker)
  * status derivation (stopped → pulling → downloading → loading → ready/error)

Both engines are fed by the SAME `_args.build_args()` output, so the vLLM flags
are identical whether a model runs in a container or as a child process.

Everything heavy (CUDA, torch, vLLM) lives inside the container/subprocess —
this module only shells out to `docker` / `vllm` / `nvidia-smi`.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import _args  # local: load() + build_args()

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "models.local.json"
RUN_DIR = ROOT / ".run"
PORT_START = 8001
CONTAINER_PREFIX = "vlm-"
STATE_VERSION = 1

# Default serving params for a freshly-pasted model (mirror models.yaml defaults).
PARAM_DEFAULTS: dict[str, Any] = {
    "max_model_len": 16384,
    "gpu_memory_util": 0.90,
    "tensor_parallel": 1,
    "max_images": 5,
    "quantization": None,
    "trust_remote_code": False,
    "mm_processor_kwargs": None,
    "hf_overrides": None,
    "extra_args": [],
}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


def hf_home() -> str:
    return os.path.expanduser(_env("HF_HOME", os.path.expanduser("~/.cache/huggingface")))


def vllm_image() -> str:
    defaults, _ = _args.load()
    return _env("VLLM_IMAGE", defaults.get("image", "vllm/vllm-openai:latest"))


# ---- llama.cpp (GGUF / ternary Bonsai) --------------------------------------
# Known locations to auto-discover a `llama-server` binary. The project's own
# build (from ./setup_llamacpp.sh, the PrismML prism fork with ternary Q2_0
# support) is preferred; the bonsai1bit_test demo build is an older fallback that
# handles 1-bit Bonsai but NOT current ternary GGUFs.
_LLAMACPP_KNOWN = [
    ROOT / ".llamacpp/llama.cpp/build/bin/llama-server",
    Path.home() / "Desktop/bonsai1bit_test/Bonsai-demo/llama.cpp/build-cuda/bin/llama-server",
    Path.home() / "Desktop/bonsai1bit_test/Bonsai-demo/bin/cuda/llama-server",
]


def llamacpp_bin() -> str | None:
    """Resolve a llama-server binary: $LLAMACPP_SERVER_BIN → known builds → PATH."""
    env = _env("LLAMACPP_SERVER_BIN")
    if env and Path(env).exists():
        return env
    for p in _LLAMACPP_KNOWN:
        if p.exists():
            return str(p)
    return shutil.which("llama-server")


def llamacpp_lib_dir() -> str:
    """Dir for LD_LIBRARY_PATH (bundled .so's live next to the binary by default)."""
    d = _env("LLAMACPP_LIB_DIR")
    if d:
        return d
    b = llamacpp_bin()
    return str(Path(b).resolve().parent) if b else ""


def llama_cache() -> str:
    """Where llama-server caches downloaded GGUFs. Kept under HF_HOME so the
    dashboard's cache library can see them alongside HF-cached models."""
    return _env("LLAMA_CACHE") or os.path.join(hf_home(), "llama.cpp")


def lan_ip() -> str:
    """Best-effort primary LAN IP (the address other machines reach us on).

    Opens a UDP socket toward a public address — no packets are actually sent;
    this just makes the OS pick the outbound interface so we can read its IP.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return ""


def public_host() -> str:
    """Host that other machines use to reach this server.

    Priority: PUBLIC_HOST / ADVERTISE_HOST override → detected LAN IP → localhost.
    (RunPod is handled separately in Manager.base_url via the pod proxy.)
    """
    return _env("PUBLIC_HOST") or _env("ADVERTISE_HOST") or lan_ip() or "localhost"


# --------------------------------------------------------------------- HF analysis
# Pipeline tags / config keys that indicate a vision (multimodal) model.
_VL_TAGS = {"image-text-to-text", "visual-question-answering",
            "image-to-text", "video-text-to-text", "any-to-any"}


def unified_memory() -> dict:
    """Total / available memory in GB.

    On the GB10 (and other unified-memory parts) the GPU shares system LPDDR, so
    /proc/meminfo is the right headroom signal — nvidia-smi reports VRAM as N/A.
    On discrete-GPU boxes this is system RAM (a coarse upper bound); the real
    limit there is VRAM, noted as a caveat in the UI.
    """
    total = avail = 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) // 1024 // 1024  # kB → GB
            elif line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) // 1024 // 1024
    except OSError:
        pass
    return {"total_gb": total, "available_gb": avail}


def _dir_size_gb(path: Path) -> float:
    """Sum on-disk size of a HF cache repo. Snapshot entries are symlinks into
    blobs/, so counting only non-symlink files avoids double-counting."""
    total = 0
    for p in path.rglob("*"):
        try:
            st = p.lstat()
        except OSError:
            continue
        # regular files only (skip the snapshot symlinks, which point into blobs/)
        if not p.is_symlink() and st.st_mode & 0o170000 == 0o100000:
            total += st.st_size
    return round(total / 1e9, 1)


def list_cached_models() -> list[dict]:
    """Enumerate already-downloaded HuggingFace repos that are runnable LLMs.

    Reads each repo's LOCAL config.json (no network) and keeps only generative
    text/vision LLMs (skips tokenizers, embeddings, ASR/TTS, classifiers, etc.).
    These can be run directly — the weights are already on disk, so launching
    one loads from cache instead of downloading.
    """
    hub = Path(hf_home()) / "hub"
    out: list[dict] = []
    if not hub.is_dir():
        return out
    for d in sorted(hub.glob("models--*")):
        hf_id = d.name[len("models--"):].replace("--", "/")
        cfgs = list(d.glob("snapshots/*/config.json"))
        if not cfgs:
            continue
        try:
            cfg = json.loads(cfgs[0].read_text())
        except (OSError, ValueError):
            continue
        archs = cfg.get("architectures") or []
        # vision = a real vision tower (not just a nested text_config, which many
        # text MoE/LLMs also have). runnable = a causal-LM or a vision LM; this
        # excludes ASR/TTS (Whisper/CSM are ForConditionalGeneration but not chat).
        multimodal = "vision_config" in cfg
        is_causal = any("ForCausalLM" in a for a in archs)
        if not (multimodal or is_causal):
            continue
        out.append({
            "hf_id": hf_id,
            "size_gb": _dir_size_gb(d),
            "multimodal": bool(multimodal),
            "arch": (cfg.get("architectures") or [None])[0],
        })
    return out


def is_cached(hf_id: str) -> bool:
    """True if the model's weight files are already present in the HF cache —
    i.e. a launch will load from disk, not re-download."""
    try:
        hf_id = normalize_hf_id(hf_id)
    except ValueError:
        return False
    snaps = Path(hf_home()) / "hub" / ("models--" + hf_id.replace("/", "--")) / "snapshots"
    if snaps.is_dir():
        for snap in snaps.iterdir():
            for ext in ("*.safetensors", "*.bin", "*.gguf", "*.pt"):
                if any(snap.glob(ext)):
                    return True
    # GGUF downloaded by llama-server lives flat in LLAMA_CACHE as
    # "<repo with / → _>_<file>.gguf".
    lc = Path(llama_cache())
    if lc.is_dir() and any(lc.glob(f"{hf_id.replace('/', '_')}_*.gguf")):
        return True
    return False


def delete_cached_model(hf_id: str) -> bool:
    """Delete a model's weights from the HF cache to free disk. Returns True if removed."""
    import shutil as _sh
    hf_id = normalize_hf_id(hf_id)
    d = Path(hf_home()) / "hub" / ("models--" + hf_id.replace("/", "--"))
    if d.is_dir():
        _sh.rmtree(d, ignore_errors=True)
        return True
    return False


def _hf_get(url: str, timeout: float = 12.0):
    """GET a HuggingFace API/JSON URL with optional HF_TOKEN; None on failure."""
    headers = {"User-Agent": "miniclosedai-llm"}
    token = _env("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = Request(url, headers=headers, method="GET")
        with urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return None
            return json.loads(r.read().decode())
    except (URLError, OSError, ValueError, TimeoutError):
        return None


def _bytes_per_param(dtype: str | None, tags: list[str]) -> float:
    t = " ".join(tags).lower()
    if any(q in t for q in ("4bit", "int4", "awq", "gptq", "-4bit")):
        return 0.5
    if "8bit" in t or "int8" in t:
        return 1.0
    d = (dtype or "").upper()
    if d.startswith("F32") or d.startswith("FP32"):
        return 4.0
    if "F8" in d or "FP8" in d or d.startswith("I8"):
        return 1.0
    if d.startswith("I4") or d.startswith("U4"):
        return 0.5
    return 2.0  # BF16 / F16 default


def _repo_tree(hf_id: str) -> list[tuple[str, int]]:
    """Return [(path, size_bytes)] for a repo, from the HF tree API."""
    tree = _hf_get(f"https://huggingface.co/api/models/{hf_id}/tree/main?recursive=true")
    if not isinstance(tree, list):
        return []
    out = []
    for e in tree:
        p = e.get("path", "")
        s = (e.get("lfs") or {}).get("size") or e.get("size") or 0
        out.append((p, int(s or 0)))
    return out


def _tree_weight_gb(hf_id: str) -> float | None:
    """Sum weight-file sizes from the repo tree (fallback when no safetensors meta)."""
    exts = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
    total = sum(s for p, s in _repo_tree(hf_id) if p.endswith(exts))
    return total / 1e9 if total else None


def pick_gguf(files: list[tuple[str, int]]) -> tuple[str, int] | None:
    """Choose which GGUF file to serve from a repo's [(path,size)] list.

    Prefers a low-bit quant over full precision; for a repo with several quants
    it favors the canonical ternary/1-bit or a sensible general default. Users can
    override with an explicit file. Returns (basename, size_bytes) or None.
    """
    ggufs = [(p.rsplit("/", 1)[-1], s) for p, s in files if p.lower().endswith(".gguf")]
    if not ggufs:
        return None
    if len(ggufs) == 1:
        return ggufs[0]
    full = ("f16.gguf", "f32.gguf", "bf16.gguf")
    pool = [(n, s) for n, s in ggufs if not any(n.lower().endswith(x) for x in full)] or ggufs
    # preference by exact suffix, most-specific first (ternary Q2_0, then common quants)
    prefs = ("-q2_0.gguf", "-q1_0.gguf", ".q4_k_m.gguf", "-q4_k_m.gguf",
             "q4_k_m.gguf", "q4_0.gguf", "q5_k_m.gguf", "q6_k.gguf", "q8_0.gguf")
    for pref in prefs:
        cands = sorted((c for c in pool if c[0].lower().endswith(pref)),
                       key=lambda c: (len(c[0]), c[0]))
        if cands:
            return cands[0]
    # fallback: smallest, then shortest name
    return sorted(pool, key=lambda c: (c[1], len(c[0])))[0]


def analyze_model(hf_id: str) -> dict:
    """Inspect a HF repo before downloading: existence, gating, type, size, fit."""
    try:
        hf_id = normalize_hf_id(hf_id)
    except ValueError as e:
        return {"exists": False, "hf_id": hf_id, "error": str(e)}

    info = _hf_get(f"https://huggingface.co/api/models/{hf_id}")
    mem = unified_memory()
    token_present = bool(_env("HF_TOKEN"))
    if info is None:
        return {"exists": False, "hf_id": hf_id, "hf_token_present": token_present,
                "available_gb": mem["available_gb"], "total_gb": mem["total_gb"],
                "error": "Not found on HuggingFace — check the id, or it may be "
                         "gated/private and need a valid HF_TOKEN in .env."}

    pipeline = info.get("pipeline_tag") or ""
    tags = [str(t) for t in (info.get("tags") or [])]
    gated = bool(info.get("gated"))

    # GGUF detection: a repo with .gguf files and no safetensors → serve via
    # llama.cpp (covers ternary Bonsai and any other GGUF). vLLM can't load these.
    tree = _repo_tree(hf_id)
    has_safetensors = any(p.endswith(".safetensors") for p, _ in tree)
    gguf = pick_gguf(tree)
    is_gguf = bool(gguf) and not has_safetensors
    gguf_file = gguf[0] if is_gguf else None

    # size: GGUF → the chosen file; else safetensors param count; else tree sum.
    st = info.get("safetensors") or {}
    params = st.get("total")
    dtype = None
    if isinstance(st.get("parameters"), dict) and st["parameters"]:
        dtype = max(st["parameters"], key=st["parameters"].get)
    if is_gguf:
        size_gb = gguf[1] / 1e9
        dtype = "GGUF"
    elif params:
        size_gb = params * _bytes_per_param(dtype, tags) / 1e9
    else:
        size_gb = _tree_weight_gb(hf_id)

    # multimodal? pipeline tag, tags, or a vision_config in config.json.
    cfg = _hf_get(f"https://huggingface.co/{hf_id}/resolve/main/config.json") or {}
    multimodal = (pipeline in _VL_TAGS
                  or any(t in _VL_TAGS for t in tags)
                  or "vision_config" in cfg)

    # Model's native context window — used to cap max_model_len so we never ask
    # vLLM for more than the model supports (which aborts startup). VLMs often
    # nest the text config under text_config.
    tcfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    max_ctx = (tcfg.get("max_position_embeddings") or cfg.get("max_position_embeddings")
               or cfg.get("max_model_len") or cfg.get("n_positions"))

    # is it even a text/generative LLM vLLM can serve? (warn on obvious non-LLMs)
    text_gen = pipeline in ("", "text-generation", "text2text-generation",
                            "conversational") or multimodal

    need_gb = round(size_gb * 1.15 + 1.0, 1) if size_gb else None
    fits = bool(need_gb and need_gb <= mem["available_gb"])

    return {
        "exists": True, "hf_id": hf_id,
        "pipeline_tag": pipeline, "multimodal": multimodal,
        "is_llm": text_gen or is_gguf, "gated": gated, "hf_token_present": token_present,
        "params": params, "dtype": dtype, "max_ctx": max_ctx,
        "fmt": "gguf" if is_gguf else "safetensors",
        "engine_hint": "llamacpp" if is_gguf else None,
        "gguf_file": gguf_file,
        "size_gb": round(size_gb, 1) if size_gb else None,
        "need_gb": need_gb, "available_gb": mem["available_gb"],
        "total_gb": mem["total_gb"], "fits": fits,
    }


# --------------------------------------------------------------------------- helpers
def normalize_hf_id(raw: str) -> str:
    """Accept `owner/name` or a full hf.co URL → canonical `owner/name`."""
    s = (raw or "").strip()
    if s.startswith(("http://", "https://", "hf.co", "huggingface.co", "www.")):
        s = re.sub(r"^[a-z]+://", "", s)
        s = re.sub(r"^(www\.)?(huggingface\.co|hf\.co)/", "", s)
    s = s.split("?")[0].split("#")[0]
    # Drop /tree/main, /blob/..., trailing slash, .git
    parts = [p for p in s.split("/") if p]
    if len(parts) >= 2:
        s = f"{parts[0]}/{parts[1]}"
    s = s.removesuffix(".git")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", s):
        raise ValueError(
            f"'{raw}' is not a valid HuggingFace repo id. Expected 'owner/name' "
            "or a https://huggingface.co/owner/name URL."
        )
    return s


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:40] or "model"


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def _run(argv: list[str], timeout: int = 20) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError:
        # Binary not on PATH (e.g. docker/nvidia-smi absent on this host). Return
        # a synthetic non-zero result so callers degrade gracefully instead of the
        # whole control plane crashing — engine availability is surfaced elsewhere.
        return subprocess.CompletedProcess(argv, 127, "", f"{argv[0]}: not found")


def probe_models(port: int, timeout: float = 2.0) -> tuple[bool, list[str]]:
    """GET http://127.0.0.1:<port>/v1/models — the authoritative readiness signal."""
    try:
        req = Request(f"http://127.0.0.1:{port}/v1/models", method="GET")
        with urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return False, []
            data = json.loads(r.read().decode())
            return True, [m.get("id", "") for m in data.get("data", [])]
    except (URLError, OSError, ValueError, TimeoutError):
        return False, []


# --------------------------------------------------------------------------- registry
@dataclass
class ModelEntry:
    id: str
    hf_id: str
    served_name: str
    port: int
    source: str = "custom"           # "preset" | "custom"
    params: dict = field(default_factory=lambda: dict(PARAM_DEFAULTS))
    desired_state: str = "stopped"   # "running" | "stopped"
    engine: str = ""                 # which engine last launched it
    multimodal: bool = True          # accepts images (gates the vision test panel)
    size_gb: float = 0.0             # reported weight footprint (for display)
    fmt: str = "safetensors"         # "safetensors" (vLLM) | "gguf" (llama.cpp)
    gguf_file: str = ""              # for gguf: which file in the repo to serve
    error: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "hf_id": self.hf_id, "served_name": self.served_name,
            "port": self.port, "source": self.source, "params": self.params,
            "desired_state": self.desired_state, "engine": self.engine,
            "multimodal": self.multimodal, "size_gb": self.size_gb,
            "fmt": self.fmt, "gguf_file": self.gguf_file,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(d: dict) -> "ModelEntry":
        params = dict(PARAM_DEFAULTS)
        params.update(d.get("params") or {})
        return ModelEntry(
            id=d["id"], hf_id=d["hf_id"], served_name=d["served_name"],
            port=int(d["port"]), source=d.get("source", "custom"), params=params,
            desired_state=d.get("desired_state", "stopped"),
            engine=d.get("engine", ""), multimodal=d.get("multimodal", True),
            size_gb=d.get("size_gb", 0.0), fmt=d.get("fmt", "safetensors"),
            gguf_file=d.get("gguf_file", ""), created_at=d.get("created_at", ""),
        )


# --------------------------------------------------------------------------- engines
class Engine:
    """Launch backend interface. Implementations: DockerEngine, NativeEngine."""

    name = "base"

    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def launch(self, e: ModelEntry) -> None:
        raise NotImplementedError

    def stop(self, e: ModelEntry) -> None:
        raise NotImplementedError

    def is_alive(self, e: ModelEntry) -> bool:
        raise NotImplementedError

    def state(self, e: ModelEntry) -> str:
        """Coarse runtime state: 'running' | 'exited' | 'absent'."""
        return "running" if self.is_alive(e) else "absent"

    def recent_logs(self, e: ModelEntry, lines: int = 60) -> str:
        raise NotImplementedError

    def open_log_stream(self, e: ModelEntry) -> subprocess.Popen | None:
        """Return a Popen whose stdout yields log lines (follow mode)."""
        raise NotImplementedError

    def discover(self) -> list[dict]:
        """Find live instances this engine owns (for startup reconcile)."""
        return []

    def _vllm_args(self, e: ModelEntry) -> list[str]:
        defaults, _ = _args.load()
        m = {"hf_id": e.hf_id, "served_name": e.served_name, "port": e.port, **e.params}
        api_key = _env("VLLM_API_KEY") or None
        return _args.build_args(defaults, m, api_key=api_key)


class DockerEngine(Engine):
    name = "docker"

    def container(self, e: ModelEntry) -> str:
        return f"{CONTAINER_PREFIX}{e.served_name}"

    def _log(self, e: ModelEntry) -> Path:
        return RUN_DIR / f"{e.served_name}.docker.log"

    def available(self) -> tuple[bool, str]:
        if not shutil.which("docker"):
            return False, "docker CLI not found on PATH"
        r = _run(["docker", "info"], timeout=15)
        if r.returncode != 0:
            return False, "docker daemon not reachable (is the user in the 'docker' group?)"
        return True, "docker daemon reachable"

    def _inspect_status(self, e: ModelEntry) -> str | None:
        r = _run(["docker", "inspect", "-f", "{{.State.Status}}", self.container(e)], timeout=15)
        return r.stdout.strip() if r.returncode == 0 else None

    def state(self, e: ModelEntry) -> str:
        s = self._inspect_status(e)
        if s is None:
            return "absent"
        return "running" if s == "running" else "exited"

    def launch(self, e: ModelEntry) -> None:
        """Non-blocking: pull the image (streamed to a log) then `docker run -d`.

        The first run downloads a multi-GB image, which is far longer than any
        sane subprocess timeout — so we detach a small bash orchestrator that
        writes pull progress + the run result to the model's log file, and
        return immediately. Status is then derived from the container state.
        """
        RUN_DIR.mkdir(exist_ok=True)
        name = self.container(e)
        image = vllm_image()
        token = _env("HF_TOKEN")
        run_argv = [
            "docker", "run", "-d", "--name", name,
            "--gpus", "all", "--ipc=host", "--shm-size", "16g",
            "-p", f"{e.port}:{e.port}",
            "-e", f"HF_TOKEN={token}",
            "-e", f"HUGGING_FACE_HUB_TOKEN={token}",
            "-e", "HF_HOME=/root/.cache/huggingface",
            "-v", f"{hf_home()}:/root/.cache/huggingface",
            "--label", "miniclosedai.manager=1",
            "--label", f"miniclosedai.served={e.served_name}",
            "--label", f"miniclosedai.port={e.port}",
            "--label", f"miniclosedai.hf_id={e.hf_id}",
            "--entrypoint", "vllm", image,
            "serve", *self._vllm_args(e),
        ]
        inner = " ".join(shlex.quote(x) for x in run_argv)
        script = (
            f"echo '== ensuring image {image} (first run downloads several GB; reused after) =='\n"
            f"docker pull {shlex.quote(image)} 2>&1\n"
            f"docker rm -f {shlex.quote(name)} >/dev/null 2>&1 || true\n"
            f"echo '== starting container =='\n"
            f"if {inner}; then echo '== container started =='; "
            f"else echo 'MANAGER-ERROR: docker run failed (see messages above)'; fi\n"
        )
        logf = self._log(e).open("wb")  # truncate: fresh log per launch
        subprocess.Popen(["bash", "-c", script], stdout=logf,
                         stderr=subprocess.STDOUT, start_new_session=True)

    def stop(self, e: ModelEntry) -> None:
        _run(["docker", "rm", "-f", self.container(e)], timeout=60)

    def is_alive(self, e: ModelEntry) -> bool:
        return self._inspect_status(e) == "running"

    def recent_logs(self, e: ModelEntry, lines: int = 60) -> str:
        # Once the container exists, its own logs are the truth; before that
        # (image pull / pre-create), read the orchestrator's log file.
        if self._inspect_status(e) is not None:
            r = _run(["docker", "logs", "--tail", str(lines), self.container(e)], timeout=15)
            return (r.stdout or "") + (r.stderr or "")
        p = self._log(e)
        return "\n".join(p.read_text(errors="replace").splitlines()[-lines:]) if p.exists() else ""

    def open_log_stream(self, e: ModelEntry) -> subprocess.Popen | None:
        if self._inspect_status(e) is not None:
            return subprocess.Popen(
                ["docker", "logs", "-f", "--tail", "400", self.container(e)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        # pre-container (pulling): follow the orchestrator log file
        self._log(e).touch(exist_ok=True)
        return subprocess.Popen(
            ["tail", "-n", "400", "-F", str(self._log(e))],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def discover(self) -> list[dict]:
        fmt = ('{{.Names}}\t{{.Label "miniclosedai.served"}}\t'
               '{{.Label "miniclosedai.port"}}\t{{.Label "miniclosedai.hf_id"}}\t{{.State}}')
        r = _run(["docker", "ps", "-a", "--filter", "label=miniclosedai.manager=1",
                  "--format", fmt], timeout=15)
        out = []
        for line in (r.stdout or "").splitlines():
            cols = line.split("\t")
            if len(cols) >= 5 and cols[4] == "running":
                out.append({"served": cols[1], "port": int(cols[2] or 0),
                            "hf_id": cols[3]})
        return out


class NativeEngine(Engine):
    name = "native"

    def _meta(self, e: ModelEntry) -> Path:
        return RUN_DIR / f"{e.served_name}.json"

    def _log(self, e: ModelEntry) -> Path:
        return RUN_DIR / f"{e.served_name}.log"

    def available(self) -> tuple[bool, str]:
        if shutil.which("vllm"):
            return True, "vllm CLI found"
        try:
            __import__("vllm")
            return True, "vllm importable"
        except Exception:
            return False, "vLLM not installed (pip install vllm) in this environment"

    def launch(self, e: ModelEntry) -> None:
        RUN_DIR.mkdir(exist_ok=True)
        vllm = shutil.which("vllm")
        cmd = [vllm, "serve", *self._vllm_args(e)] if vllm \
            else ["python", "-m", "vllm.entrypoints.cli.main", "serve", *self._vllm_args(e)]
        logf = self._log(e).open("wb")
        env = dict(os.environ)
        token = _env("HF_TOKEN")
        if token:
            env["HF_TOKEN"] = token
            env["HUGGING_FACE_HUB_TOKEN"] = token
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True, env=env,  # own process group → killpg later
        )
        self._meta(e).write_text(json.dumps(
            {"pid": proc.pid, "port": e.port, "served": e.served_name,
             "hf_id": e.hf_id, "engine": self.name}
        ))

    def _pid(self, e: ModelEntry) -> int | None:
        try:
            return int(json.loads(self._meta(e).read_text())["pid"])
        except Exception:
            return None

    def stop(self, e: ModelEntry) -> None:
        pid = self._pid(e)
        if pid:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(pid), sig)
                except ProcessLookupError:
                    break
                except OSError:
                    break
                time.sleep(0.5)
                if not self._alive_pid(pid):
                    break
        self._meta(e).unlink(missing_ok=True)

    @staticmethod
    def _alive_pid(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def is_alive(self, e: ModelEntry) -> bool:
        pid = self._pid(e)
        return bool(pid and self._alive_pid(pid))

    def state(self, e: ModelEntry) -> str:
        pid = self._pid(e)
        if pid is None:
            return "absent"               # never launched / cleanly stopped
        return "running" if self._alive_pid(pid) else "exited"

    def recent_logs(self, e: ModelEntry, lines: int = 60) -> str:
        p = self._log(e)
        if not p.exists():
            return ""
        data = p.read_text(errors="replace").splitlines()
        return "\n".join(data[-lines:])

    def open_log_stream(self, e: ModelEntry) -> subprocess.Popen | None:
        # tail -F follows by name and tolerates the file not existing yet.
        RUN_DIR.mkdir(exist_ok=True)
        self._log(e).touch(exist_ok=True)
        return subprocess.Popen(
            ["tail", "-n", "400", "-F", str(self._log(e))],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )

    def discover(self) -> list[dict]:
        out = []
        if not RUN_DIR.is_dir():
            return out
        for meta in RUN_DIR.glob("*.json"):
            try:
                d = json.loads(meta.read_text())
            except Exception:
                continue
            if d.get("pid") and self._alive_pid(int(d["pid"])):
                out.append({"served": d.get("served", meta.stem),
                            "port": int(d.get("port", 0)), "hf_id": d.get("hf_id", ""),
                            "engine": d.get("engine", self.name)})
        return out


class LlamaCppEngine(NativeEngine):
    """Serve GGUF models (ternary Bonsai etc.) via a native `llama-server`.

    Inherits NativeEngine's pid-file/log/state/stop machinery; only the launch
    command differs. llama-server is OpenAI-compatible (`/v1/models` +
    `/v1/chat/completions`), so readiness probing, base_url, and the whole
    register-in-miniclosedai flow work unchanged.
    """
    name = "llamacpp"

    def available(self) -> tuple[bool, str]:
        b = llamacpp_bin()
        if b:
            return True, f"llama-server: {b}"
        return False, "llama-server not found — run ./setup_llamacpp.sh (ternary needs the PrismML fork)"

    def launch(self, e: ModelEntry) -> None:
        RUN_DIR.mkdir(exist_ok=True)
        binp = llamacpp_bin()
        if not binp:
            raise RuntimeError("no llama-server binary; run ./setup_llamacpp.sh")
        cmd = [binp, "--hf-repo", e.hf_id]
        if e.gguf_file:
            cmd += ["--hf-file", e.gguf_file]
        cmd += ["--host", "0.0.0.0", "--port", str(e.port), "-ngl", "99", "-c", "0",
                "--jinja"]
        key = _env("VLLM_API_KEY")
        if key:
            cmd += ["--api-key", key]
        cmd += list(e.params.get("extra_args") or [])

        env = dict(os.environ)
        lib = llamacpp_lib_dir()
        if lib:  # bundled .so's next to the binary
            env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        env["LLAMA_CACHE"] = llama_cache()
        os.makedirs(env["LLAMA_CACHE"], exist_ok=True)
        token = _env("HF_TOKEN")
        if token:
            env["HF_TOKEN"] = token
            env["HUGGING_FACE_HUB_TOKEN"] = token

        logf = self._log(e).open("wb")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
        self._meta(e).write_text(json.dumps(
            {"pid": proc.pid, "port": e.port, "served": e.served_name,
             "hf_id": e.hf_id, "engine": self.name}))


# --------------------------------------------------------------------------- manager
class Manager:
    def __init__(self) -> None:
        self.entries: dict[str, ModelEntry] = {}
        self.docker = DockerEngine()
        self.native = NativeEngine()
        self.llamacpp = LlamaCppEngine()   # GGUF / ternary Bonsai
        self.engine = self._select_engine()
        self._add_lock = threading.Lock()  # serialize add() so concurrent
        # duplicate submits can't both create an entry (the "-2" duplicate bug)

    # ---- engine selection -------------------------------------------------
    def _select_engine(self) -> Engine:
        choice = _env("LAUNCH_ENGINE", "auto").lower()
        if choice == "docker":
            return self.docker
        if choice == "native":
            return self.native
        # auto: prefer docker when its daemon is reachable, else native.
        if self.docker.available()[0]:
            return self.docker
        if self.native.available()[0]:
            return self.native
        return self.docker  # degraded; surfaced via engine_info()

    def engine_info(self) -> dict:
        d_ok, d_msg = self.docker.available()
        n_ok, n_msg = self.native.available()
        l_ok, l_msg = self.llamacpp.available()
        gpu = self.gpu_info()
        return {
            "engine": self.engine.name,
            "engine_override": _env("LAUNCH_ENGINE", "auto"),
            "docker_ok": d_ok, "docker_msg": d_msg,
            "native_ok": n_ok, "native_msg": n_msg,
            "llamacpp_ok": l_ok, "llamacpp_msg": l_msg,
            "gpu_ok": bool(gpu.get("gpus")),
            "image": vllm_image(),
            "hf_home": hf_home(),
            "runpod": bool(_env("RUNPOD_POD_ID")),
            "lan_ip": lan_ip(),
            "public_host": public_host(),
            "no_engine": not (d_ok or n_ok),
        }

    @staticmethod
    def gpu_info() -> dict:
        if not shutil.which("nvidia-smi"):
            return {"gpus": [], "error": "nvidia-smi not found"}
        r = _run(["nvidia-smi",
                  "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                  "--format=csv,noheader,nounits"], timeout=10)
        if r.returncode != 0:
            return {"gpus": [], "error": (r.stderr or "nvidia-smi failed").strip()}
        def _num(x):
            # GB10 / unified-memory parts report "[N/A]" for VRAM fields.
            try:
                return int(float(x))
            except (TypeError, ValueError):
                return None
        gpus = []
        for line in r.stdout.strip().splitlines():
            c = [x.strip() for x in line.split(",")]
            if len(c) >= 5:
                gpus.append({"index": _num(c[0]) or 0, "name": c[1],
                             "mem_total_mb": _num(c[2]),
                             "mem_used_mb": _num(c[3]),
                             "util_pct": _num(c[4]) or 0})
        return {"gpus": gpus}

    # ---- persistence ------------------------------------------------------
    def load(self) -> None:
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                for d in data.get("models", []):
                    e = ModelEntry.from_dict(d)
                    self.entries[e.id] = e
            except Exception:
                pass

    def save(self) -> None:
        STATE_FILE.write_text(json.dumps(
            {"version": STATE_VERSION,
             "models": [e.to_dict() for e in self.entries.values()]}, indent=2))

    def _seed_presets(self) -> None:
        defaults, models = _args.load()
        for m in models:
            sid = m["served_name"]
            # one card per served_name; skip if a custom/preset already claims it
            if sid in self.entries:
                continue
            if any(e.served_name == sid for e in self.entries.values()):
                continue
            self.entries[sid] = ModelEntry(
                id=sid, hf_id=m["hf_id"], served_name=sid, port=int(m["port"]),
                source="preset",
                params={
                    "max_model_len": m.get("max_model_len", PARAM_DEFAULTS["max_model_len"]),
                    "gpu_memory_util": m.get("gpu_memory_util", PARAM_DEFAULTS["gpu_memory_util"]),
                    "tensor_parallel": m.get("tensor_parallel", 1),
                    "max_images": m.get("max_images", 5),
                    "quantization": m.get("quantization"),
                    "trust_remote_code": bool(m.get("trust_remote_code")),
                    "mm_processor_kwargs": m.get("mm_processor_kwargs"),
                    "hf_overrides": m.get("hf_overrides"),
                    "extra_args": list(m.get("extra_args") or []),
                },
                desired_state="stopped",
                created_at=_now(),
            )

    def reconcile(self) -> None:
        """Load registry, seed presets, re-attach to live instances."""
        self.load()
        self._seed_presets()
        # Re-attach: anything the active engine reports running is marked running.
        # Any entry previously marked running but NOT live (container/process gone,
        # or its launch died with a prior manager) is reset to stopped — otherwise
        # it would be stuck showing "pulling"/"loading" forever. We don't
        # auto-relaunch; the user clicks Run again.
        # Discover across ALL engines (docker containers + native/llamacpp procs);
        # native.discover() already covers llamacpp (same .run/*.json, tagged by
        # engine), so docker + native is the full set.
        live = {}
        for d in self.docker.discover() + self.native.discover():
            live.setdefault(d["served"], d)
        for e in self.entries.values():
            if e.served_name in live:
                d = live[e.served_name]
                e.desired_state = "running"
                e.engine = d.get("engine") or e.engine or self.engine.name
                if d.get("port"):
                    e.port = d["port"]
            elif e.desired_state == "running":
                e.desired_state = "stopped"
        # Live instances with no registry entry → synthesize a card so the user
        # can see/stop them (e.g. registry file was deleted).
        for served, d in live.items():
            if not any(e.served_name == served for e in self.entries.values()):
                self.entries[served] = ModelEntry(
                    id=served, hf_id=d.get("hf_id", "") or "(unknown)",
                    served_name=served, port=d.get("port") or 0,
                    source="custom", desired_state="running",
                    engine=d.get("engine") or self.engine.name,
                    fmt="gguf" if d.get("engine") == "llamacpp" else "safetensors",
                    created_at=_now())
        self.save()

    # ---- allocation -------------------------------------------------------
    def _unique_served(self, base: str) -> str:
        name, i = base, 2
        existing = {e.served_name for e in self.entries.values()}
        while name in existing:
            name = f"{base}-{i}"
            i += 1
        return name

    def next_free_port(self, start: int = PORT_START) -> int:
        used = {e.port for e in self.entries.values()}
        p = start
        while p in used or not _port_is_free(p):
            p += 1
        return p

    # ---- CRUD + lifecycle -------------------------------------------------
    def add(self, hf_id: str, served_name: str | None = None,
            port: int | None = None, params: dict | None = None,
            run: bool = True, force: bool = False) -> ModelEntry:
        hf_id = normalize_hf_id(hf_id)

        # Analyze the repo first: existence, type (text vs multimodal), fit.
        report = analyze_model(hf_id)
        if not report.get("exists"):
            raise ValueError(report.get("error", "model not found on HuggingFace"))
        if not force and report.get("fits") is False and report.get("need_gb"):
            err = ValueError(
                f"{hf_id} needs ~{report['need_gb']} GB but only "
                f"{report['available_gb']} GB is free. Pass force to run anyway, "
                f"or pick a smaller / quantized model.")
            err.analysis = report  # type: ignore[attr-defined]
            raise err

        # GGUF → llama.cpp engine; everything else → vLLM (docker/native).
        is_gguf = report.get("fmt") == "gguf"
        engine_name = "llamacpp" if is_gguf else self.engine.name
        gguf_file = report.get("gguf_file") if is_gguf else ""
        if (params or {}).get("gguf_file"):
            gguf_file = params["gguf_file"]

        merged = dict(PARAM_DEFAULTS)
        merged.update(params or {})
        merged.pop("gguf_file", None)  # not a vLLM/serving param
        multimodal = bool(report.get("multimodal"))

        if not is_gguf:
            # --- vLLM-only param massaging (skipped for GGUF/llama.cpp) ---
            # Cap max_model_len to the model's native context — asking vLLM for
            # more than the model supports aborts startup (SmolLM2 is 8192).
            if not (params or {}).get("max_model_len") and report.get("max_ctx"):
                merged["max_model_len"] = min(int(merged["max_model_len"]), int(report["max_ctx"]))
            # Size gpu_memory_util from FREE memory, not total (GB10 unified mem).
            if not (params or {}).get("gpu_memory_util") and report.get("total_gb"):
                avail, total = report["available_gb"], report["total_gb"]
                merged["gpu_memory_util"] = max(0.30, round(min(0.90, avail * 0.85 / total), 2))
            # Text-only LLMs must NOT get image flags.
            if not multimodal:
                if not (params or {}).get("max_images"):
                    merged["max_images"] = None
                merged.setdefault("mm_processor_kwargs", None)
            _validate_params(merged)

        # Critical section: reject a duplicate of an already-active model and
        # allocate name/port atomically, so two near-simultaneous submits (a
        # double-clicked button, two tabs) can't both create an entry.
        with self._add_lock:
            dup = next((x for x in self.entries.values()
                        if x.hf_id == hf_id and x.desired_state != "stopped"), None)
            if dup is not None:
                raise ValueError(
                    f"{hf_id} is already running as '{dup.served_name}' on port "
                    f"{dup.port}. Stop it first to launch another copy.")
            served = slugify(served_name) if served_name else slugify(hf_id.split("/")[-1])
            served = self._unique_served(served)
            if port is None:
                port = self.next_free_port()
            elif any(x.port == port for x in self.entries.values()):
                raise ValueError(f"port {port} is already assigned to another model")
            # Mark running up-front (when launching) so a concurrent submit sees
            # it as active and is rejected by the dup check above.
            e = ModelEntry(id=served, hf_id=hf_id, served_name=served, port=port,
                           source="custom", params=merged, engine=engine_name,
                           desired_state="running" if run else "stopped",
                           multimodal=multimodal, size_gb=report.get("size_gb") or 0.0,
                           fmt=report.get("fmt", "safetensors"), gguf_file=gguf_file or "",
                           created_at=_now())
            self.entries[e.id] = e
            self.save()
        if run:
            self.start(e.id)
        return e

    def get(self, mid: str) -> ModelEntry:
        if mid not in self.entries:
            raise KeyError(mid)
        return self.entries[mid]

    def start(self, mid: str) -> ModelEntry:
        e = self.get(mid)
        eng = self._engine_for(e)
        ok, msg = eng.available()
        if not ok:
            raise RuntimeError(f"launch engine '{eng.name}' unavailable: {msg}")
        e.error = ""
        try:
            eng.launch(e)
            e.desired_state = "running"
            e.engine = eng.name
        except Exception as exc:
            e.error = str(exc)[-2000:]
            e.desired_state = "stopped"
            self.save()
            raise
        self.save()
        return e

    def stop(self, mid: str) -> ModelEntry:
        e = self.get(mid)
        self._engine_for(e).stop(e)
        e.desired_state = "stopped"
        self.save()
        return e

    def remove(self, mid: str) -> None:
        e = self.get(mid)
        try:
            self.stop(mid)
        except Exception:
            pass
        del self.entries[mid]
        self.save()

    # ---- status + views ---------------------------------------------------
    def _engine_for(self, e: ModelEntry) -> Engine:
        if e.engine == "docker":
            return self.docker
        if e.engine == "native":
            return self.native
        if e.engine == "llamacpp":
            return self.llamacpp
        return self.engine

    @staticmethod
    def _scan_error(logs: str) -> str | None:
        low = logs.lower()
        markers = ("out of memory", "manager-error", "no matching manifest",
                   "401 client error", "403 client error", "gatedrepoerror",
                   "repositorynotfounderror", "traceback (most recent call last)",
                   "error response from daemon",
                   # llama.cpp / GGUF failures
                   "invalid ggml type", "failed to load model", "unknown model architecture",
                   "failed to read tensor", "error: failed to download")
        return next((m for m in markers if m in low), None)

    def derive_status(self, e: ModelEntry) -> dict:
        eng = self._engine_for(e)
        st = eng.state(e)

        if st == "running":
            ready, ids = probe_models(e.port)
            if ready:
                return {"status": "ready", "ready": True, "detail": ", ".join(ids)}
            logs = eng.recent_logs(e, 100)
            if self._scan_error(logs):
                return {"status": "error", "ready": False, "detail": eng.recent_logs(e, 40)}
            low = logs.lower()
            # If the weights are already cached, this phase is load/compile, NOT a
            # download — never mislabel it "downloading" (vLLM's weight-loading
            # progress bar also prints "%|", which would otherwise fool us).
            if not is_cached(e.hf_id) and any(
                    k in low for k in ("downloading", "fetching", "resolving data")):
                status = "downloading"
            else:
                status = "loading"
            return {"status": status, "ready": False, "detail": ""}

        if st == "exited":
            # container/process started then died → surface the tail as the error.
            return {"status": "error", "ready": False,
                    "detail": (eng.recent_logs(e, 40) or e.error or "exited")}

        # absent: either mid-launch (pulling image) or genuinely stopped.
        if e.desired_state == "running":
            logs = eng.recent_logs(e, 100)
            if self._scan_error(logs) or e.error:
                return {"status": "error", "ready": False, "detail": logs or e.error}
            return {"status": "pulling", "ready": False, "detail": ""}
        if e.error:
            return {"status": "error", "ready": False, "detail": e.error[-1200:]}
        return {"status": "stopped", "ready": False, "detail": ""}

    def base_url(self, e: ModelEntry, public: bool = True) -> str:
        """URL to register in miniclosedai.

        - not public  → 127.0.0.1 (used by the in-app vision-test proxy only).
        - RunPod      → the pod's public proxy URL.
        - otherwise   → the LAN IP (so a miniclosedai on another machine, or this
                        host's, can reach it), overridable via PUBLIC_HOST.
        """
        if not public:
            return f"http://127.0.0.1:{e.port}/v1"
        host_override = _env("PUBLIC_HOST") or _env("ADVERTISE_HOST")
        pod = _env("RUNPOD_POD_ID")
        if pod and not host_override:
            return f"https://{pod}-{e.port}.proxy.runpod.net/v1"
        return f"http://{host_override or lan_ip() or 'localhost'}:{e.port}/v1"

    def alt_base_url(self, e: ModelEntry) -> str:
        """Alternative for when miniclosedai runs as a Docker container on THIS
        same host — it reaches the server via the docker host gateway."""
        return f"http://host.docker.internal:{e.port}/v1"

    def view(self, e: ModelEntry) -> dict:
        st = self.derive_status(e)
        return {
            **e.to_dict(),
            "status": st["status"],
            "ready": st["ready"],
            "detail": st["detail"],
            "error": e.error,
            "base_url": self.base_url(e),
            "alt_base_url": self.alt_base_url(e),
            "local_url": self.base_url(e, public=False),
            "container": f"{CONTAINER_PREFIX}{e.served_name}",
        }

    def list_views(self) -> list[dict]:
        return [self.view(e) for e in sorted(
            self.entries.values(), key=lambda x: (x.source != "custom", x.served_name))]

    def open_log_stream(self, mid: str) -> subprocess.Popen | None:
        e = self.get(mid)
        return self._engine_for(e).open_log_stream(e)


def _now() -> str:
    # Avoid Date.now-style nondeterminism concerns are irrelevant here (server side).
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _validate_params(p: dict) -> None:
    for jf in ("mm_processor_kwargs", "hf_overrides"):
        v = p.get(jf)
        if v:
            try:
                json.loads(v)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{jf} must be valid JSON: {exc}")
    if not (0.1 <= float(p["gpu_memory_util"]) <= 1.0):
        raise ValueError("gpu_memory_util must be between 0.1 and 1.0")
    if int(p["max_model_len"]) < 256:
        raise ValueError("max_model_len too small")
    if isinstance(p.get("extra_args"), str):
        p["extra_args"] = p["extra_args"].split()
