#!/usr/bin/env python3
"""miniclosedai-llm — web control plane.

A FastAPI app + static GUI that lets you paste a HuggingFace vision-language
model id and have it downloaded + served behind an OpenAI `/v1` API (via vLLM),
with live status/logs, a built-in image test, and a copy-able base_url to
register in the miniclosedai gateway.

Heavy lifting (CUDA/torch/vLLM) runs inside the launched container/subprocess;
this app only orchestrates via `model_manager`. Binds 0.0.0.0:MANAGER_PORT
(default 8099). Mirrors the sibling miniclosedai-voice server (SSE bridge,
_NoCacheStatics, optional Bearer auth).
"""
from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path

import httpx
from fastapi import (Depends, FastAPI, File, Form, Header, HTTPException,
                     Request, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import model_manager as mm

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
TEST_IMAGE = ROOT / "tests" / "test_image.png"
API_KEY = mm._env("MANAGER_API_KEY")
PORT = int(mm._env("MANAGER_PORT", "8099"))
VERSION = "1.0.0"

manager = mm.Manager()


def _require_auth(authorization: str | None = Header(None)) -> None:
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "Invalid or missing Authorization header.")


app = FastAPI(title="miniclosedai-llm control plane", docs_url="/docs", redoc_url=None)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    await asyncio.to_thread(manager.reconcile)


# --------------------------------------------------------------------------- schemas
class AddModelRequest(BaseModel):
    hf_id: str
    served_name: str | None = None
    port: int | None = None
    params: dict | None = None
    run: bool = True
    force: bool = False


class AnalyzeRequest(BaseModel):
    hf_id: str


class HFTokenRequest(BaseModel):
    token: str


# --------------------------------------------------------------------------- meta
@app.get("/api/health")
async def health(_=Depends(_require_auth)):
    info = await asyncio.to_thread(manager.engine_info)
    host = info.get("lan_ip") or "localhost"
    info["manager_port"] = PORT
    info["dashboard_url"] = f"http://{host}:{PORT}"
    return {"ok": True, "version": VERSION, **info}


@app.get("/api/gpu")
async def gpu(_=Depends(_require_auth)):
    return await asyncio.to_thread(manager.gpu_info)


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest, _=Depends(_require_auth)):
    return await asyncio.to_thread(mm.analyze_model, req.hf_id)


@app.get("/api/hf-token")
async def hf_token_get(_=Depends(_require_auth)):
    """Is a Hugging Face token configured? (masked — never returns the secret)."""
    return await asyncio.to_thread(mm.hf_token_status)


@app.post("/api/hf-token")
async def hf_token_set(req: HFTokenRequest, _=Depends(_require_auth)):
    """Save a pasted token — applies immediately (next launch/retry) and to .env."""
    res = await asyncio.to_thread(mm.set_hf_token, req.token)
    if not res.get("ok"):
        raise HTTPException(400, res.get("error", "could not set token"))
    return res


@app.delete("/api/hf-token")
async def hf_token_clear(_=Depends(_require_auth)):
    await asyncio.to_thread(mm.clear_hf_token)
    return {"ok": True}


@app.get("/api/cache")
async def cache(_=Depends(_require_auth)):
    """Already-downloaded LLMs in the HF cache — runnable without re-downloading."""
    models = await asyncio.to_thread(mm.list_cached_models)
    return {"models": models, "hf_home": mm.hf_home(),
            "total_gb": round(sum(m["size_gb"] for m in models), 1)}


class CacheDeleteRequest(BaseModel):
    hf_id: str


@app.post("/api/cache/delete")
async def cache_delete(req: CacheDeleteRequest, _=Depends(_require_auth)):
    removed = await asyncio.to_thread(mm.delete_cached_model, req.hf_id)
    if not removed:
        raise HTTPException(404, "not found in cache")
    return {"ok": True}


@app.get("/api/test-image")
async def test_image(_=Depends(_require_auth)):
    if not TEST_IMAGE.exists():
        raise HTTPException(404, "test image missing")
    return FileResponse(str(TEST_IMAGE), media_type="image/png")


# --------------------------------------------------------------------------- models CRUD
@app.get("/api/models")
async def list_models(_=Depends(_require_auth)):
    return {"models": await asyncio.to_thread(manager.list_views)}


@app.post("/api/models", status_code=201)
async def add_model(req: AddModelRequest, _=Depends(_require_auth)):
    try:
        e = await asyncio.to_thread(
            manager.add, req.hf_id, req.served_name, req.port, req.params,
            req.run, req.force)
    except ValueError as exc:
        # Doesn't-fit errors carry the analysis so the UI can offer "Run anyway".
        analysis = getattr(exc, "analysis", None)
        if analysis is not None:
            raise HTTPException(409, {"message": str(exc), "analysis": analysis})
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    return await asyncio.to_thread(manager.view, e)


@app.post("/api/models/{mid}/start")
async def start_model(mid: str, _=Depends(_require_auth)):
    try:
        e = await asyncio.to_thread(manager.start, mid)
    except KeyError:
        raise HTTPException(404, "no such model")
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    return await asyncio.to_thread(manager.view, e)


@app.post("/api/models/{mid}/stop")
async def stop_model(mid: str, _=Depends(_require_auth)):
    try:
        e = await asyncio.to_thread(manager.stop, mid)
    except KeyError:
        raise HTTPException(404, "no such model")
    return await asyncio.to_thread(manager.view, e)


@app.delete("/api/models/{mid}")
async def delete_model(mid: str, _=Depends(_require_auth)):
    try:
        await asyncio.to_thread(manager.remove, mid)
    except KeyError:
        raise HTTPException(404, "no such model")
    return {"ok": True}


@app.get("/api/models/{mid}/status")
async def model_status(mid: str, _=Depends(_require_auth)):
    try:
        e = manager.get(mid)
    except KeyError:
        raise HTTPException(404, "no such model")
    return await asyncio.to_thread(manager.derive_status, e)


# --------------------------------------------------------------------------- logs (SSE)
@app.get("/api/models/{mid}/logs")
async def model_logs(mid: str, request: Request, _=Depends(_require_auth)):
    try:
        entry = manager.get(mid)
    except KeyError:
        raise HTTPException(404, "no such model")

    async def gen():
        try:
            proc = await asyncio.to_thread(manager.open_log_stream, mid)
        except Exception as exc:  # engine/log source unavailable
            yield _sse({"eof": True, "detail": str(exc)})
            return
        if proc is None:
            yield _sse({"eof": True, "detail": "no log stream available"})
            return
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        loop = asyncio.get_running_loop()
        SENTINEL = object()

        def _pump():
            try:
                for line in proc.stdout:  # type: ignore[union-attr]
                    asyncio.run_coroutine_threadsafe(
                        q.put(("line", line.rstrip("\n"))), loop)
            finally:
                asyncio.run_coroutine_threadsafe(q.put(SENTINEL), loop)

        worker = asyncio.create_task(asyncio.to_thread(_pump))
        last_probe = 0.0
        try:
            # emit an initial status frame immediately
            st = await asyncio.to_thread(manager.derive_status, entry)
            yield _sse({"status": st["status"], "ready": st["ready"]})
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(q.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    item = None
                if item is SENTINEL:
                    yield _sse({"eof": True})
                    break
                if item is not None:
                    yield _sse({"line": item[1]})
                now = time.monotonic()
                if now - last_probe > 2.0:
                    last_probe = now
                    st = await asyncio.to_thread(manager.derive_status, entry)
                    yield _sse({"status": st["status"], "ready": st["ready"],
                                "needs_hf_token": st.get("needs_hf_token", False)})
        finally:
            try:
                proc.terminate()
            except Exception:
                pass
            worker.cancel()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


# --------------------------------------------------------------------------- vision test
@app.post("/api/models/{mid}/test")
async def test_model(mid: str, request: Request,
                     prompt: str = Form("Say hello and briefly introduce yourself."),
                     max_tokens: int = Form(300),
                     image: UploadFile | None = File(None),
                     _=Depends(_require_auth)):
    try:
        entry = manager.get(mid)
    except KeyError:
        raise HTTPException(404, "no such model")

    # Text test by default; attach an image only if one was uploaded (multimodal).
    if image is not None:
        raw = await image.read()
        mime = image.content_type or "image/png"
        data_url = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
        content = [{"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": data_url}}]
    else:
        content = prompt  # plain-text chat — works for any LLM

    payload = {
        "model": entry.served_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    vkey = mm._env("VLLM_API_KEY")
    if vkey:
        headers["Authorization"] = f"Bearer {vkey}"
    url = manager.base_url(entry, public=False).rstrip("/") + "/chat/completions"
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"could not reach model on port {entry.port}: {exc}")
    latency_ms = int((time.monotonic() - t0) * 1000)
    if r.status_code != 200:
        raise HTTPException(502, f"model returned HTTP {r.status_code}: {r.text[:500]}")
    body = r.json()
    answer = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return {"answer": answer, "usage": body.get("usage"), "latency_ms": latency_ms}


# --------------------------------------------------------------------------- static (last)
class _NoCacheStatics(StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp


if STATIC_DIR.is_dir():
    app.mount("/", _NoCacheStatics(directory=str(STATIC_DIR), html=True), name="static")


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):  # pragma: no cover
    return JSONResponse(status_code=500, content={"detail": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, log_level="info")
