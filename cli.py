#!/usr/bin/env python3
"""mc — terminal client for the miniclosedai-llm control plane.

Everything the web dashboard does, from the shell: analyze / run / stop / test /
chat with HuggingFace LLMs. It's a thin HTTP client over the same `/api` endpoints
the GUI uses, so the two stay in live sync (run a model here → it shows in the
browser, and vice-versa).

Dependency-free: standard library only (argparse + urllib + json) — runs under any
python3, no venv required. Talks to the dashboard at $MANAGER_URL
(default http://localhost:$MANAGER_PORT, 8099). Sends Authorization: Bearer
$MANAGER_API_KEY when that env var is set.

Run `mc <command> -h` for per-command help. Common commands:
    mc info | ls | gpu | cache
    mc analyze <hf_id>
    mc run <hf_id> [--wait] [--name N --port P --quant Q --gpu-mem F ...]
    mc test <id> "your prompt"          mc chat <id>
    mc logs <id> [-f]                    mc url <id>
    mc stop <id> | rm <id> | free <hf_id>
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXIT_OK, EXIT_ERR, EXIT_UNREACHABLE = 0, 1, 2


# --------------------------------------------------------------------- config
def _load_dotenv() -> dict:
    env = {}
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


_DOTENV = _load_dotenv()


def cfg(name: str, default: str = "") -> str:
    return os.environ.get(name) or _DOTENV.get(name) or default


def base_url() -> str:
    url = cfg("MANAGER_URL")
    if url:
        return url.rstrip("/")
    return f"http://localhost:{cfg('MANAGER_PORT', '8099')}"


def _headers(extra: dict | None = None) -> dict:
    h = {"Accept": "application/json"}
    key = cfg("MANAGER_API_KEY")
    if key:
        h["Authorization"] = f"Bearer {key}"
    if extra:
        h.update(extra)
    return h


# --------------------------------------------------------------------- ANSI
_TTY = sys.stdout.isatty()


def c(text, color):
    if not _TTY:
        return text
    codes = {"dim": "2", "red": "31", "green": "32", "yellow": "33",
             "blue": "34", "cyan": "36", "bold": "1"}
    return f"\033[{codes[color]}m{text}\033[0m"


STATUS_COLOR = {"ready": "green", "error": "red", "stopped": "dim",
                "pulling": "yellow", "downloading": "yellow", "loading": "yellow"}


# --------------------------------------------------------------------- HTTP
class ApiError(Exception):
    def __init__(self, status, detail):
        self.status = status
        self.detail = detail
        msg = detail.get("message") if isinstance(detail, dict) else detail
        super().__init__(msg if isinstance(msg, str) else json.dumps(msg))


class Unreachable(Exception):
    pass


def _request(method, path, *, data=None, headers=None, timeout=60):
    url = path if path.startswith("http") else base_url() + path
    req = urllib.request.Request(url, data=data, method=method,
                                 headers=_headers(headers))
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            detail = json.loads(body).get("detail", body)
        except ValueError:
            detail = body
        raise ApiError(e.code, detail)
    except (urllib.error.URLError, ConnectionError, socket.timeout, OSError) as e:
        raise Unreachable(str(e))


def api_get(path, timeout=30):
    with _request("GET", path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def api_post(path, obj=None, timeout=120):
    data = json.dumps(obj or {}).encode()
    with _request("POST", path, data=data,
                  headers={"Content-Type": "application/json"}, timeout=timeout) as r:
        body = r.read().decode()
        return json.loads(body) if body else None


def api_delete(path, timeout=60):
    with _request("DELETE", path, timeout=timeout) as r:
        body = r.read().decode()
        return json.loads(body) if body else None


def api_multipart(path, fields, files, timeout=300):
    boundary = "----mc" + str(int(time.time() * 1000))
    parts = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    for k, (fn, content, ctype) in files.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fn}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n".encode() + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    with _request("POST", path, data=body,
                  headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                  timeout=timeout) as r:
        return json.loads(r.read().decode())


# --------------------------------------------------------------------- helpers
def die(msg, code=EXIT_ERR):
    print(c("error:", "red") + " " + msg, file=sys.stderr)
    sys.exit(code)


def require_daemon():
    """Friendly check that the dashboard is up; exits 2 if not."""
    try:
        return api_get("/api/health", timeout=8)
    except Unreachable:
        die(f"dashboard not running at {base_url()} — start it:  ./dev.sh   (or  ./mc serve)",
            EXIT_UNREACHABLE)
    except ApiError as e:
        if e.status in (401, 403):
            die("unauthorized — set MANAGER_API_KEY to match the dashboard.", EXIT_UNREACHABLE)
        raise


def resolve_id(key: str) -> str:
    """Resolve a model reference to an exact served-name id (forgiving)."""
    models = api_get("/api/models")["models"]
    ids = [m["id"] for m in models]
    if key in ids:
        return key
    pref = [i for i in ids if i.startswith(key)]
    if len(pref) == 1:
        return pref[0]
    byhf = [m["id"] for m in models if key.lower() in m["hf_id"].lower()]
    if len(byhf) == 1:
        return byhf[0]
    if not models:
        die("no models yet — run one first:  mc run <hf_id>")
    cands = pref or byhf or ids
    die(f"'{key}' didn't match one model. Candidates: {', '.join(cands)}")


def _table(rows, headers):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(c(line, "bold"))
    for r in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(r)))


# --------------------------------------------------------------------- commands
def cmd_info(args):
    h = require_daemon()
    eng = h.get("engine")
    print(f"{c('engine', 'dim')}     {eng}  ({'GPU ok' if h.get('gpu_ok') else c('no GPU', 'yellow')})")
    print(f"{c('dashboard', 'dim')}  {h.get('dashboard_url')}")
    print(f"{c('hf_home', 'dim')}    {h.get('hf_home')}")
    if h.get("no_engine"):
        print(c("  no launch engine available — install Docker or `pip install vllm`", "red"))
    try:
        g = api_get("/api/gpu")
        for gpu in g.get("gpus", []):
            mem = "unified memory" if gpu["mem_total_mb"] is None else f"{gpu['mem_used_mb']}/{gpu['mem_total_mb']} MB"
            print(f"{c('gpu', 'dim')}        GPU{gpu['index']} {gpu['name']} — {mem} ({gpu['util_pct']}%)")
        if g.get("error"):
            print(f"{c('gpu', 'dim')}        {c(g['error'], 'yellow')}")
    except ApiError:
        pass


def cmd_gpu(args):
    require_daemon()
    g = api_get("/api/gpu")
    if args.json:
        print(json.dumps(g, indent=2)); return
    if not g.get("gpus"):
        print(c("no GPU: " + (g.get("error") or "not detected"), "yellow")); return
    for gpu in g["gpus"]:
        mem = "unified" if gpu["mem_total_mb"] is None else f"{gpu['mem_used_mb']}/{gpu['mem_total_mb']} MB"
        print(f"GPU{gpu['index']}  {gpu['name']}  {mem}  util {gpu['util_pct']}%")


def cmd_ls(args):
    require_daemon()
    models = api_get("/api/models")["models"]
    if args.json:
        print(json.dumps(models, indent=2)); return
    if not models:
        print(c("no models — run one:  mc run <hf_id>", "dim")); return
    rows = []
    for m in sorted(models, key=lambda x: (x["source"] != "custom", x["id"])):
        st = c(m["status"], STATUS_COLOR.get(m["status"], "dim"))
        kind = "gguf" if m.get("fmt") == "gguf" else ("vision" if m.get("multimodal") else "text")
        rows.append([m["id"], st, f":{m['port']}", kind, m["hf_id"]])
    _table(rows, ["NAME", "STATUS", "PORT", "KIND", "HF_ID"])


def cmd_analyze(args):
    require_daemon()
    a = api_post("/api/analyze", {"hf_id": args.hf_id})
    if args.json:
        print(json.dumps(a, indent=2)); return
    if not a.get("exists"):
        die(a.get("error", "not found"))
    typ = "vision+text" if a["multimodal"] else "text"
    print(f"{c(a['hf_id'], 'bold')}  ({typ})")
    if a.get("fmt") == "gguf":
        print(f"  format     GGUF → llama.cpp" + (f"  ·  {a['gguf_file']}" if a.get("gguf_file") else ""))
    if a.get("params"):
        print(f"  params     {a['params']/1e9:.1f} B" + (f" · {a['dtype']}" if a.get('dtype') else ""))
    if a.get("size_gb") is not None:
        print(f"  weights    ~{a['size_gb']} GB")
    if a.get("need_gb") is not None:
        print(f"  needs      ~{a['need_gb']} GB")
    print(f"  available  {a['available_gb']} GB / {a['total_gb']} GB total")
    if a.get("gated"):
        print("  gated      " + ("yes (HF_TOKEN set)" if a["hf_token_present"]
                                  else c("yes — set HF_TOKEN in .env", "yellow")))
    verdict = c("fits ✓", "green") if a.get("fits") else c("may not fit ⚠", "yellow")
    print(f"  verdict    {verdict}")


def _run_params(args) -> dict:
    p = {}
    if args.max_len is not None: p["max_model_len"] = args.max_len
    if args.gpu_mem is not None: p["gpu_memory_util"] = args.gpu_mem
    if args.tp is not None: p["tensor_parallel"] = args.tp
    if args.max_images is not None: p["max_images"] = args.max_images
    if args.quant: p["quantization"] = args.quant
    if args.trust_remote_code: p["trust_remote_code"] = True
    if getattr(args, "gguf_file", None): p["gguf_file"] = args.gguf_file
    return p


def cmd_run(args):
    require_daemon()
    body = {"hf_id": args.hf_id, "run": True, "force": args.force}
    if args.name: body["served_name"] = args.name
    if args.port: body["port"] = args.port
    p = _run_params(args)
    if p: body["params"] = p
    try:
        m = api_post("/api/models", body)
    except ApiError as e:
        # 409 doesn't-fit carries the analysis; show the message + how to override
        if e.status == 409:
            msg = e.detail.get("message") if isinstance(e.detail, dict) else e.detail
            die(f"{msg}\n  re-run with --force to launch anyway.")
        die(str(e))
    print(f"launching {c(m['served_name'], 'bold')} on :{m['port']}  "
          f"({'vision' if m.get('multimodal') else 'text'}, {m.get('size_gb', '?')} GB)")
    if not args.wait:
        print(c(f"  watch:  mc logs {m['id']} -f", "dim"))
        return
    _wait_ready(m["id"], args.timeout)


def _wait_ready(mid, timeout):
    start = time.time()
    last = None
    while time.time() - start < timeout:
        st = api_get(f"/api/models/{mid}/status")
        s = st["status"]
        if s != last:
            print(f"  [{int(time.time()-start):4d}s] {c(s, STATUS_COLOR.get(s, 'dim'))}")
            last = s
        if st.get("ready"):
            print(c(f"ready — try:  mc test {mid} \"hello\"   or   mc chat {mid}", "green"))
            return
        if s == "error":
            die("failed to start:\n" + (st.get("detail") or "")[-800:])
        time.sleep(3)
    die(f"timed out after {timeout}s (last: {last})")


def cmd_start(args):
    require_daemon()
    mid = resolve_id(args.id)
    try:
        m = api_post(f"/api/models/{mid}/start")
    except ApiError as e:
        die(str(e))
    print(f"starting {c(mid, 'bold')} on :{m['port']}")
    if args.wait:
        _wait_ready(mid, args.timeout)
    else:
        print(c(f"  watch:  mc logs {mid} -f", "dim"))


def cmd_stop(args):
    require_daemon()
    mid = resolve_id(args.id)
    api_post(f"/api/models/{mid}/stop")
    print(f"stopped {mid}")


def cmd_rm(args):
    require_daemon()
    mid = resolve_id(args.id)
    api_delete(f"/api/models/{mid}")
    print(f"removed {mid}  (weights kept; free them with: mc free <hf_id>)")


def cmd_status(args):
    require_daemon()
    mid = resolve_id(args.id)
    st = api_get(f"/api/models/{mid}/status")
    if args.json:
        print(json.dumps(st, indent=2)); return
    print(f"{mid}: {c(st['status'], STATUS_COLOR.get(st['status'], 'dim'))}"
          + (f"  ({st['detail']})" if st.get("detail") else ""))


def cmd_logs(args):
    require_daemon()
    mid = resolve_id(args.id)
    # SSE stream. With -f, follow until Ctrl-C/eof. Without -f, print a snapshot
    # and stop — bounded by a wall-clock deadline (an actively-serving container
    # logs continuously, so an idle read-timeout alone wouldn't end it).
    read_timeout = None if args.follow else 5.0
    deadline = None if args.follow else time.time() + 6.0
    try:
        resp = _request("GET", f"/api/models/{mid}/logs", timeout=read_timeout)
    except Unreachable as e:
        die(str(e), EXIT_UNREACHABLE)
    try:
        for raw in resp:
            if deadline and time.time() > deadline:
                break
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                d = json.loads(line[5:].strip())
            except ValueError:
                continue
            if "line" in d:
                print(d["line"])
            elif d.get("eof"):
                break
    except (socket.timeout, TimeoutError):
        pass  # idle drain complete (non-follow)
    except KeyboardInterrupt:
        pass
    finally:
        resp.close()


def cmd_test(args):
    require_daemon()
    mid = resolve_id(args.id)
    fields = {"prompt": args.prompt, "max_tokens": str(args.max_tokens)}
    files = {}
    if args.image:
        p = Path(args.image)
        if not p.exists():
            die(f"image not found: {args.image}")
        files["image"] = (p.name, p.read_bytes(), "image/png")
    try:
        r = api_multipart(f"/api/models/{mid}/test", fields, files)
    except ApiError as e:
        die(str(e))
    print(r.get("answer", "").strip())
    u = r.get("usage") or {}
    print(c(f"\n[{r.get('latency_ms')} ms"
            + (f" · {u.get('total_tokens')} tok" if u.get('total_tokens') else "") + "]", "dim"))


def cmd_url(args):
    require_daemon()
    mid = resolve_id(args.id)
    m = next(x for x in api_get("/api/models")["models"] if x["id"] == mid)
    print(f"Kind:     openai")
    print(f"Base URL: {c(m['base_url'], 'cyan')}")
    print(c(f"(same-host Docker miniclosedai? use {m['alt_base_url']})", "dim"))


def cmd_cache(args):
    require_daemon()
    if args.sub == "rm":
        api_post("/api/cache/delete", {"hf_id": args.hf_id})
        print(f"freed {args.hf_id} from cache"); return
    d = api_get("/api/cache")
    if args.json:
        print(json.dumps(d, indent=2)); return
    models = d.get("models", [])
    if not models:
        print(c("no downloaded LLMs in the cache yet.", "dim")); return
    rows = [[m["hf_id"], f"{m['size_gb']} GB", "vision" if m["multimodal"] else "text"]
            for m in sorted(models, key=lambda x: -x["size_gb"])]
    _table(rows, ["HF_ID", "SIZE", "KIND"])
    print(c(f"\n{len(models)} models · {d.get('total_gb')} GB in {d.get('hf_home')}", "dim"))
    print(c("run one (loads from disk, no re-download):  mc run <hf_id>", "dim"))


def cmd_free(args):
    require_daemon()
    try:
        api_post("/api/cache/delete", {"hf_id": args.hf_id})
    except ApiError as e:
        die(str(e))
    print(f"freed {args.hf_id} from cache")


def cmd_chat(args):
    require_daemon()
    mid = resolve_id(args.id)
    m = next(x for x in api_get("/api/models")["models"] if x["id"] == mid)
    if not m.get("ready"):
        die(f"{mid} is '{m['status']}', not ready. Start it: mc run {m['hf_id']} --wait")
    port = m["port"]
    chat_url = f"http://localhost:{port}/v1/chat/completions"
    vkey = cfg("VLLM_API_KEY")
    headers = {"Content-Type": "application/json"}
    if vkey:
        headers["Authorization"] = f"Bearer {vkey}"
    history = []
    print(c(f"chatting with {mid} (:{port}). /reset clears history, /exit quits.", "dim"))
    while True:
        try:
            prompt = input(c("you> ", "cyan"))
        except (EOFError, KeyboardInterrupt):
            print(); break
        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt in ("/exit", "/quit", "/q"):
            break
        if prompt == "/reset":
            history.clear(); print(c("(history cleared)", "dim")); continue
        history.append({"role": "user", "content": prompt})
        payload = {"model": mid, "messages": history, "stream": True,
                   "temperature": args.temperature, "max_tokens": args.max_tokens}
        req = urllib.request.Request(chat_url, data=json.dumps(payload).encode(),
                                     method="POST", headers=headers)
        sys.stdout.write(c("bot> ", "green"))
        sys.stdout.flush()
        acc = ""
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for raw in resp:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        delta = json.loads(chunk)["choices"][0]["delta"].get("content", "")
                    except (ValueError, KeyError, IndexError):
                        continue
                    if delta:
                        acc += delta
                        sys.stdout.write(delta); sys.stdout.flush()
        except urllib.error.HTTPError as e:
            print(c(f"\n[model error {e.code}: {e.read().decode()[:200]}]", "red")); history.pop(); continue
        except (urllib.error.URLError, KeyboardInterrupt) as e:
            print(c(f"\n[interrupted: {e}]", "yellow")); history.pop(); continue
        print()
        history.append({"role": "assistant", "content": acc})


def cmd_serve(args):
    dev = ROOT / "dev.sh"
    if not dev.exists():
        die("dev.sh not found")
    os.execv("/bin/bash", ["bash", str(dev)])


# --------------------------------------------------------------------- parser
def build_parser():
    p = argparse.ArgumentParser(
        prog="mc", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("info", help="engine + GPU + dashboard status").set_defaults(fn=cmd_info)

    g = sub.add_parser("gpu", help="GPU readout"); g.add_argument("--json", action="store_true"); g.set_defaults(fn=cmd_gpu)

    for name in ("ls", "list"):
        s = sub.add_parser(name, help="list models"); s.add_argument("--json", action="store_true"); s.set_defaults(fn=cmd_ls)

    s = sub.add_parser("analyze", help="inspect a HF model before downloading")
    s.add_argument("hf_id"); s.add_argument("--json", action="store_true"); s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("run", help="download + run a model")
    s.add_argument("hf_id")
    s.add_argument("--name", help="served model name (default: slug of repo)")
    s.add_argument("--port", type=int)
    s.add_argument("--quant", help="fp8 | awq | gptq")
    s.add_argument("--max-len", type=int, dest="max_len")
    s.add_argument("--gpu-mem", type=float, dest="gpu_mem")
    s.add_argument("--tp", type=int, help="tensor parallel size (#GPUs)")
    s.add_argument("--max-images", type=int, dest="max_images")
    s.add_argument("--trust-remote-code", action="store_true")
    s.add_argument("--gguf-file", dest="gguf_file", help="for a multi-file GGUF repo, pick the file")
    s.add_argument("--force", action="store_true", help="launch even if it may not fit")
    s.add_argument("--wait", action="store_true", help="poll until ready")
    s.add_argument("--timeout", type=int, default=1200)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("start", help="(re)start an existing stopped model by id")
    s.add_argument("id"); s.add_argument("--wait", action="store_true")
    s.add_argument("--timeout", type=int, default=1200); s.set_defaults(fn=cmd_start)

    s = sub.add_parser("stop", help="stop a running model"); s.add_argument("id"); s.set_defaults(fn=cmd_stop)
    s = sub.add_parser("rm", help="remove a model (keeps weights)"); s.add_argument("id"); s.set_defaults(fn=cmd_rm)
    s = sub.add_parser("status", help="status of one model"); s.add_argument("id"); s.add_argument("--json", action="store_true"); s.set_defaults(fn=cmd_status)

    s = sub.add_parser("logs", help="stream a model's logs"); s.add_argument("id")
    s.add_argument("-f", "--follow", action="store_true", help="follow (Ctrl-C to stop)"); s.set_defaults(fn=cmd_logs)

    s = sub.add_parser("test", help="one-shot chat test")
    s.add_argument("id"); s.add_argument("prompt", nargs="?", default="Say hello in one short sentence.")
    s.add_argument("--image"); s.add_argument("--max-tokens", type=int, default=300, dest="max_tokens")
    s.set_defaults(fn=cmd_test)

    s = sub.add_parser("chat", help="interactive chat REPL")
    s.add_argument("id"); s.add_argument("--max-tokens", type=int, default=1024, dest="max_tokens")
    s.add_argument("--temperature", type=float, default=0.7); s.set_defaults(fn=cmd_chat)

    s = sub.add_parser("url", help="base_url to register in miniclosedai"); s.add_argument("id"); s.set_defaults(fn=cmd_url)

    for name in ("cache", "downloaded"):
        s = sub.add_parser(name, help="list / manage already-downloaded models")
        s.add_argument("sub", nargs="?", choices=["rm"], help="rm <hf_id> to delete weights")
        s.add_argument("hf_id", nargs="?")
        s.add_argument("--json", action="store_true"); s.set_defaults(fn=cmd_cache)

    s = sub.add_parser("free", help="delete a cached model's weights"); s.add_argument("hf_id"); s.set_defaults(fn=cmd_free)

    sub.add_parser("serve", help="start the dashboard (runs ./dev.sh)").set_defaults(fn=cmd_serve)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        parser.print_help(); return EXIT_OK
    try:
        args.fn(args)
        return EXIT_OK
    except Unreachable:
        die(f"dashboard not running at {base_url()} — start it:  ./dev.sh  (or ./mc serve)",
            EXIT_UNREACHABLE)
    except ApiError as e:
        die(str(e))
    except BrokenPipeError:
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
