#!/usr/bin/env python3
"""End-to-end regression test for the miniclosedai-llm control plane.

For each small test model it drives the REAL manager API exactly like the GUI:

    add + run  →  poll until ready  →  text chat  →  (image chat if multimodal)
              →  assert sensible output  →  stop + remove

This catches the regressions we actually hit while building this:
  * model never loads / errors on this GPU (e.g. arm64 image, OOM, bad flag)
  * server returns empty output
  * the prompt is ignored and every answer is identical (the Form-field bug)
  * multimodal models don't accept images

It only uses the standard library (urllib) and talks to a running dashboard
(`./dev.sh`), so it doubles as a CI smoke test. Exits non-zero on any failure.

Usage:
  python3 e2e_test.py                       # default: small text + small vision
  python3 e2e_test.py --quick               # just the tiny text model (fast)
  python3 e2e_test.py --models A/B C/D      # test specific HF ids
  python3 e2e_test.py --base http://localhost:8099 --timeout 900 --keep
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEST_IMAGE = ROOT / "tests" / "test_image.png"

# Small models that load quickly enough for a regression run. Each is a dict:
#   hf_id, kind ("text"|"vision"), expect (substring that should appear, lower-case)
QUICK = [
    {"hf_id": "Qwen/Qwen2.5-0.5B-Instruct", "kind": "text", "expect": "paris"},
    # Different family (Llama-arch) + tiny ~720 MB download → fast integrity check
    # that also covers a non-Qwen code path.
    {"hf_id": "HuggingFaceTB/SmolLM2-360M-Instruct", "kind": "text", "expect": "paris"},
]
DEFAULT = QUICK + [
    {"hf_id": "Qwen/Qwen2.5-VL-3B-Instruct", "kind": "vision", "expect": "blue"},
]
# Ternary GGUF (llama.cpp path) — opt-in via --gguf, since it needs the PrismML
# llama.cpp fork built (./setup_llamacpp.sh). The smallest ternary Bonsai.
GGUF = [
    {"hf_id": "prism-ml/Ternary-Bonsai-1.7B-gguf", "kind": "text", "expect": "paris"},
]


# --------------------------------------------------------------------- HTTP helpers
def _req(method, url, data=None, headers=None, timeout=60):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
        return r.status, (json.loads(body) if body else None)


def get(base, path, timeout=30):
    return _req("GET", base + path, timeout=timeout)


def post_json(base, path, obj, timeout=60):
    return _req("POST", base + path, json.dumps(obj).encode(),
                {"Content-Type": "application/json"}, timeout)


def delete(base, path):
    try:
        _req("DELETE", base + path)
    except urllib.error.URLError:
        pass


def post_multipart(base, path, fields, files, timeout=120):
    """Minimal multipart/form-data POST (stdlib only)."""
    boundary = "----miniclosedaie2e" + str(int(time.time() * 1000))
    parts = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    for k, (fn, content, ctype) in files.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fn}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n".encode() + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    return _req("POST", base + path, body, headers, timeout)


# --------------------------------------------------------------------- test driver
def chat(base, mid, prompt, image=None, timeout=120):
    fields = {"prompt": prompt}
    files = {}
    if image:
        files["image"] = ("img.png", image, "image/png")
    _, body = post_multipart(base, f"/api/models/{mid}/test", fields, files, timeout)
    return (body or {}).get("answer", "")


def wait_ready(base, mid, timeout, log):
    start = time.time()
    last = None
    while time.time() - start < timeout:
        try:
            _, st = get(base, f"/api/models/{mid}/status", timeout=10)
        except urllib.error.URLError as e:
            log(f"    (status fetch failed: {e})"); time.sleep(5); continue
        s = st.get("status")
        if s != last:
            log(f"    [{int(time.time()-start):4d}s] {s}")
            last = s
        if st.get("ready"):
            return True, ""
        if s == "error":
            return False, st.get("detail", "")[-600:]
        time.sleep(5)
    return False, f"timed out after {timeout}s (last status: {last})"


def run_one(base, spec, timeout, keep, log):
    hf = spec["hf_id"]
    res = {"hf_id": hf, "kind": spec["kind"], "ok": False, "notes": []}
    t0 = time.time()
    log(f"\n=== {hf}  ({spec['kind']}) ===")

    # launch (force=true so a tight-fit small model still runs)
    try:
        code, m = post_json(base, "/api/models", {"hf_id": hf, "run": True, "force": True})
    except urllib.error.HTTPError as e:
        res["notes"].append(f"add failed: HTTP {e.code} {e.read().decode()[:200]}")
        return res
    mid = m["id"]
    log(f"  launched as '{m['served_name']}' on :{m['port']} (multimodal={m['multimodal']})")

    try:
        ready, err = wait_ready(base, mid, timeout, log)
        if not ready:
            res["notes"].append(f"not ready: {err}")
            return res

        # --- text chat: two DIFFERENT prompts must give DIFFERENT answers ----
        a1 = chat(base, mid, "What is the capital of France? Answer with only the city name.")
        a2 = chat(base, mid, "Tell me a one-line joke about computers.")
        log(f"  text#1: {a1.strip()[:80]!r}")
        log(f"  text#2: {a2.strip()[:80]!r}")
        if not a1.strip() or not a2.strip():
            res["notes"].append("empty text answer"); return res
        if a1.strip() == a2.strip():
            res["notes"].append("IDENTICAL answers to different prompts (prompt ignored!)")
            return res
        # For text models, `expect` is a soft check on the factual text answer.
        # (For vision models it applies to the image answer below.)
        if spec["kind"] == "text" and spec.get("expect") and spec["expect"] not in a1.lower():
            res["notes"].append(f"warn: expected '{spec['expect']}' not in answer")

        # --- image chat for multimodal models -------------------------------
        if spec["kind"] == "vision":
            if not TEST_IMAGE.exists():
                res["notes"].append("warn: test image missing, skipped vision check")
            else:
                ans = chat(base, mid,
                           "What is the dominant background color of this image? One word.",
                           image=TEST_IMAGE.read_bytes(), timeout=180)
                log(f"  vision: {ans.strip()[:80]!r}")
                if not ans.strip():
                    res["notes"].append("empty vision answer"); return res
                if "blue" not in ans.lower():
                    res["notes"].append("warn: vision answer didn't mention 'blue'")

        res["ok"] = True
    finally:
        res["secs"] = int(time.time() - t0)
        if not keep:
            delete(base, f"/api/models/{mid}")
            log(f"  cleaned up ({res['secs']}s)")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8099")
    ap.add_argument("--timeout", type=int, default=900, help="per-model ready timeout (s)")
    ap.add_argument("--models", nargs="+", help="HF ids to test (override defaults)")
    ap.add_argument("--quick", action="store_true", help="only the tiny text models")
    ap.add_argument("--gguf", action="store_true", help="ternary GGUF (llama.cpp) model — needs ./setup_llamacpp.sh")
    ap.add_argument("--keep", action="store_true", help="don't stop/remove after testing")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    if args.models:
        specs = [{"hf_id": h, "kind": "text", "expect": None} for h in args.models]
    elif args.gguf:
        specs = GGUF
    elif args.quick:
        specs = QUICK
    else:
        specs = DEFAULT

    def log(m): print(m, flush=True)

    # preflight: manager reachable + an engine available
    try:
        _, h = get(base, "/api/health", timeout=10)
    except urllib.error.URLError as e:
        log(f"FAILED: manager not reachable at {base} ({e}). Start it with ./dev.sh"); return 2
    log(f"manager OK · engine={h.get('engine')} · gpu_ok={h.get('gpu_ok')} · {base}")
    if h.get("no_engine"):
        log("FAILED: no launch engine available (no Docker, no native vLLM)."); return 2

    results = [run_one(base, s, args.timeout, args.keep, log) for s in specs]

    log("\n" + "=" * 64)
    log("REGRESSION SUMMARY")
    failed = 0
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        if not r["ok"]:
            failed += 1
        warns = [n for n in r["notes"] if n.startswith("warn:")]
        hard = [n for n in r["notes"] if not n.startswith("warn:")]
        log(f"  [{status}] {r['hf_id']:<38} {r.get('secs','?')}s")
        for n in hard:
            log(f"          ✗ {n}")
        for n in warns:
            log(f"          ! {n}")
    log("=" * 64)
    log(f"{len(results)-failed}/{len(results)} models passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
