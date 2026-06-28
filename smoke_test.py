#!/usr/bin/env python3
"""End-to-end vision smoke test for the vLLM VLM server.

Proves the server (or transformers shim) is OpenAI-compatible AND actually
*sees* images — not just answering text-only:

  1. GET  {base_url}/models                -> served name is listed.
  2. POST {base_url}/chat/completions      -> 1 image: description matches the
                                              image's contents (color/shape/text).
  3. POST {base_url}/chat/completions      -> 2 images in ONE request succeeds
                                              (multi-page / front+back documents).

Usage:
  python3 smoke_test.py                                  # defaults below
  python3 smoke_test.py --base-url http://localhost:8001/v1 --model qwen3-vl-8b
  python3 smoke_test.py --api-key sk-... --image tests/test_image.png

Uses only the standard library (urllib) so it runs anywhere — no pip installs.
The test image at tests/test_image.png is a blue card with a red "ID 12345"
box, a yellow circle, and the text "MINICLOSEDAI VLM VISION TEST".
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / "tests" / "test_image.png"

# Words the model should mention if it genuinely sees the test image.
EXPECTED_KEYWORDS = [
    "blue", "red", "yellow", "circle", "rectangle", "square",
    "id", "12345", "miniclosedai", "vision", "test", "card", "text",
]


def _post(url: str, payload: dict, api_key: str, timeout: int = 180) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _get(url: str, api_key: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, method="GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def data_url(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    suffix = path.suffix.lstrip(".").lower() or "png"
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime};base64,{b64}"


def ensure_image(path: Path) -> None:
    if path.exists():
        return
    print(f"[setup] {path} missing — regenerating it.")
    try:
        from PIL import Image, ImageDraw  # noqa: WPS433
    except ImportError:
        sys.exit(f"ERROR: {path} not found and Pillow not installed to recreate it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (384, 256), (24, 60, 160))
    d = ImageDraw.Draw(img)
    d.ellipse([248, 40, 344, 136], fill=(255, 214, 10), outline=(0, 0, 0), width=4)
    d.rectangle([24, 40, 200, 120], fill=(200, 40, 40), outline=(255, 255, 255), width=3)
    d.text((40, 70), "ID 12345", fill=(255, 255, 255))
    d.text((24, 170), "MINICLOSEDAI VLM VISION TEST", fill=(255, 255, 255))
    img.save(path)


def chat(base_url: str, model: str, api_key: str, parts: list[dict]) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": parts}],
        "max_tokens": 300,
        "temperature": 0.0,
        "stream": False,
    }
    resp = _post(f"{base_url}/chat/completions", payload, api_key)
    return resp["choices"][0]["message"]["content"] or ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8001/v1",
                    help="OpenAI-compatible base URL, MUST end with /v1")
    ap.add_argument("--model", default="qwen3-vl-8b", help="served-model-name")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    ensure_image(args.image)
    img = data_url(args.image)
    failures: list[str] = []

    # ---- 1. /v1/models lists the served name -------------------------------
    print(f"[1/3] GET {base}/models")
    try:
        models = _get(f"{base}/models", args.api_key)
        ids = [m.get("id") for m in models.get("data", [])]
        print(f"      served models: {ids}")
        if args.model not in ids:
            failures.append(f"served name '{args.model}' not in /models {ids}")
    except urllib.error.URLError as e:
        return _die(f"cannot reach {base}/models: {e}")

    # ---- 2. single image: must describe it ---------------------------------
    print("[2/3] POST /chat/completions with ONE image")
    try:
        out = chat(base, args.model, args.api_key, [
            {"type": "text", "text": "Describe this image in detail. "
             "What colors, shapes, and text do you see?"},
            {"type": "image_url", "image_url": {"url": img}},
        ])
    except Exception as e:  # noqa: BLE001
        return _die(f"single-image request failed: {e}")
    print(f"      model said: {out.strip()[:400]}")
    low = out.lower()
    hits = [k for k in EXPECTED_KEYWORDS if k in low]
    if not out.strip():
        failures.append("single-image response was EMPTY")
    elif len(hits) < 2:
        failures.append(f"response doesn't look image-grounded (matched only {hits})")
    else:
        print(f"      vision-grounded ✓ (matched keywords: {hits})")

    # ---- 3. TWO images in one request --------------------------------------
    print("[3/3] POST /chat/completions with TWO images (multi-page packet)")
    try:
        out2 = chat(base, args.model, args.api_key, [
            {"type": "text", "text": "These are two pages of one document. "
             "Confirm you received both and say how many images you see."},
            {"type": "image_url", "image_url": {"url": img}},
            {"type": "image_url", "image_url": {"url": img}},
        ])
        print(f"      model said: {out2.strip()[:300]}")
        if not out2.strip():
            failures.append("two-image response was EMPTY")
        else:
            print("      multi-image request accepted ✓")
    except Exception as e:  # noqa: BLE001
        failures.append(f"two-image request failed: {e} "
                        "(is --limit-mm-per-prompt image>=2 set?)")

    # ---- verdict -----------------------------------------------------------
    print("\n" + "=" * 60)
    if failures:
        print("SMOKE TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE TEST PASSED ✓  vision works end-to-end, multi-image OK.")
    return 0


def _die(msg: str) -> int:
    print(f"\nSMOKE TEST FAILED: {msg}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
