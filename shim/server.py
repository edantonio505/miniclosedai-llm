#!/usr/bin/env python3
"""OpenAI-compatible `/v1` shim backed by HuggingFace transformers.

Bare-metal fallback for any HF model that Docker/vLLM can't serve here (e.g.
Jetson aarch64). It exposes the EXACT same surface miniclosedai talks to —
`GET /v1/models` and `POST /v1/chat/completions` with multimodal `content`
arrays and base64 `image_url` data-URLs — so the gateway cannot tell it apart
from vLLM.

Vision models load via `AutoProcessor` + `AutoModelForImageTextToText` (Qwen-VL,
InternVL-HF, Llava, …); text models via `AutoTokenizer` + `AutoModelForCausalLM`
(Llama, Qwen, Mistral, …). `SHIM_MODALITY` (auto|text|vlm) picks the path; `auto`
tries the VLM classes and falls back to causal-LM.

Config is entirely via environment (no hardcoded paths/secrets):
    SHIM_MODEL_ID      HF repo id to load (default Qwen/Qwen2.5-VL-7B-Instruct)
    SHIM_SERVED_NAME   name advertised in /v1/models + accepted as `model`
    SHIM_PORT          bind port (default 8009)
    SHIM_HOST          bind host (default 0.0.0.0)
    SHIM_MAX_IMAGES    max images accepted per request (default 5)
    SHIM_API_KEY       if set, require `Authorization: Bearer <it>` (else open)
    SHIM_DTYPE         bfloat16 | float16 | auto (default bfloat16)
    SHIM_MAX_NEW_TOKENS default cap on generated tokens (default 1024)
    SHIM_TRUST_REMOTE_CODE  1/0 (default 1 — InternVL needs it)
    SHIM_MODALITY      auto | text | vlm  (default auto)
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from threading import Thread

import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    TextIteratorStreamer,
)

# --------------------------------------------------------------------------- config
MODEL_ID = os.environ.get("SHIM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
SERVED_NAME = os.environ.get("SHIM_SERVED_NAME", "qwen2.5-vl-7b")
HOST = os.environ.get("SHIM_HOST", "0.0.0.0")
PORT = int(os.environ.get("SHIM_PORT", "8009"))
MAX_IMAGES = int(os.environ.get("SHIM_MAX_IMAGES", "5"))
API_KEY = os.environ.get("SHIM_API_KEY", "") or None
DEFAULT_MAX_NEW = int(os.environ.get("SHIM_MAX_NEW_TOKENS", "1024"))
TRUST_REMOTE = os.environ.get("SHIM_TRUST_REMOTE_CODE", "1") not in ("0", "false", "")
MODALITY = os.environ.get("SHIM_MODALITY", "auto").lower()  # auto | text | vlm
_DTYPE = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "auto": "auto",
}.get(os.environ.get("SHIM_DTYPE", "bfloat16"), torch.bfloat16)

STATE: dict = {}


def _load_vlm():
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=_DTYPE, device_map="auto", trust_remote_code=TRUST_REMOTE)
    return proc, model, True


def _load_text():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=_DTYPE, device_map="auto", trust_remote_code=TRUST_REMOTE)
    return tok, model, False


def _load_model():
    """Return (processor_or_tokenizer, model, is_vlm) per SHIM_MODALITY."""
    if MODALITY == "vlm":
        return _load_vlm()
    if MODALITY == "text":
        return _load_text()
    # auto: try the generic VLM classes first (Qwen-VL, Llava, InternVL, …); if
    # the model isn't an image-text-to-text arch, transformers raises → load it
    # as a plain causal LM (Llama, Qwen, Mistral, … text models), all bare-metal.
    try:
        return _load_vlm()
    except Exception as ex:
        print(f"[shim] not a VLM ({type(ex).__name__}: {str(ex)[:120]}) "
              f"— loading as a causal LM", flush=True)
        return _load_text()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[shim] loading {MODEL_ID} as '{SERVED_NAME}' "
          f"(modality={MODALITY}, dtype={_DTYPE}) ...", flush=True)
    proc, model, is_vlm = _load_model()
    model.eval()
    STATE["processor"] = proc
    STATE["model"] = model
    STATE["is_vlm"] = is_vlm
    print(f"[shim] ready on {HOST}:{PORT}  -> serves '{SERVED_NAME}' "
          f"({'vlm' if is_vlm else 'text'})", flush=True)
    yield
    STATE.clear()


app = FastAPI(title="miniclosedai transformers shim", lifespan=lifespan)


# --------------------------------------------------------------------------- schema
class ChatRequest(BaseModel):
    model: str | None = None
    messages: list
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False


def _check_auth(authorization: str | None) -> None:
    if API_KEY is None:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid api key")


def _load_image(url: str) -> Image.Image:
    """Decode a data:image/...;base64,... URL (or plain http url) into PIL."""
    if url.startswith("data:"):
        header, _, b64 = url.partition(",")
        if ";base64" not in header:
            raise HTTPException(400, "only base64 data URLs are supported")
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if url.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    raise HTTPException(400, f"unsupported image url scheme: {url[:24]}...")


def _to_hf(messages: list) -> tuple[list, list[Image.Image]]:
    """Translate OpenAI messages -> HF chat-template messages + ordered images."""
    hf_messages: list = []
    images: list[Image.Image] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, str):
            hf_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue
        parts = []
        for p in content or []:
            ptype = p.get("type")
            if ptype == "text":
                parts.append({"type": "text", "text": p.get("text", "")})
            elif ptype == "image_url":
                url = (p.get("image_url") or {}).get("url", "")
                images.append(_load_image(url))
                parts.append({"type": "image"})
        hf_messages.append({"role": role, "content": parts})
    if len(images) > MAX_IMAGES:
        raise HTTPException(400, f"too many images: {len(images)} > SHIM_MAX_IMAGES={MAX_IMAGES}")
    return hf_messages, images


def _to_text_messages(messages: list) -> list:
    """OpenAI messages -> plain {role, content:str} for a text tokenizer's chat
    template. Image parts are rejected (this path serves text-only models)."""
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        text_parts = []
        for p in content or []:
            if p.get("type") == "text":
                text_parts.append(p.get("text", ""))
            elif p.get("type") == "image_url":
                raise HTTPException(400, "this model is text-only; image content is not supported")
        out.append({"role": role, "content": "\n".join(text_parts)})
    return out


def _prepare_inputs(messages: list):
    processor, model = STATE["processor"], STATE["model"]
    if not STATE.get("is_vlm"):
        # Text-only causal LM: render the tokenizer's chat template, then tokenize.
        prompt = processor.apply_chat_template(
            _to_text_messages(messages), tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, return_tensors="pt")
        return inputs.to(model.device)
    hf_messages, images = _to_hf(messages)
    prompt = processor.apply_chat_template(
        hf_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt],
        images=images or None,
        return_tensors="pt",
        padding=True,
    )
    return inputs.to(model.device)


def _gen_kwargs(req: ChatRequest) -> dict:
    kw: dict = {"max_new_tokens": req.max_tokens or DEFAULT_MAX_NEW}
    if req.temperature and req.temperature > 0:
        kw.update(do_sample=True, temperature=req.temperature)
        if req.top_p is not None:
            kw["top_p"] = req.top_p
    else:
        kw["do_sample"] = False
    return kw


# --------------------------------------------------------------------------- routes
@app.get("/health")
async def health():
    return {"status": "ok" if STATE.get("model") else "loading", "model": SERVED_NAME}


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    return {
        "object": "list",
        "data": [{"id": SERVED_NAME, "object": "model", "owned_by": "miniclosedai-shim",
                  "created": int(time.time())}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatRequest, authorization: str | None = Header(default=None)
):
    _check_auth(authorization)
    if STATE.get("model") is None:
        raise HTTPException(503, "model still loading")

    processor, model = STATE["processor"], STATE["model"]
    inputs = _prepare_inputs(req.messages)
    in_len = inputs["input_ids"].shape[1]
    gen_kwargs = _gen_kwargs(req)
    rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    served = req.model or SERVED_NAME

    if req.stream:
        return StreamingResponse(
            _stream(model, processor, inputs, gen_kwargs, rid, served),
            media_type="text/event-stream",
        )

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0][in_len:]
    text = processor.decode(new_tokens, skip_special_tokens=True).strip()

    return {
        "id": rid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": served,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": int(in_len),
            "completion_tokens": int(new_tokens.shape[0]),
            "total_tokens": int(in_len + new_tokens.shape[0]),
        },
    }


def _stream(model, processor, inputs, gen_kwargs, rid, served):
    streamer = TextIteratorStreamer(
        processor.tokenizer if hasattr(processor, "tokenizer") else processor,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    thread = Thread(target=_run_generate, args=(model, inputs, gen_kwargs, streamer))
    thread.start()

    def frame(delta=None, finish=None):
        choice = {"index": 0, "delta": {} if delta is None else {"content": delta},
                  "finish_reason": finish}
        return "data: " + json.dumps({
            "id": rid, "object": "chat.completion.chunk",
            "created": int(time.time()), "model": served, "choices": [choice],
        }) + "\n\n"

    yield frame(delta="")  # opening role frame
    for piece in streamer:
        if piece:
            yield frame(delta=piece)
    yield frame(finish="stop")
    yield "data: [DONE]\n\n"
    thread.join()


def _run_generate(model, inputs, gen_kwargs, streamer):
    with torch.inference_mode():
        model.generate(**inputs, **gen_kwargs, streamer=streamer)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
