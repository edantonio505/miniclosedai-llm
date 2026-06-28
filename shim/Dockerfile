# transformers VLM shim — OpenAI-compatible /v1 fallback for models vLLM can't serve.
# Built on the PyTorch CUDA runtime so GPU inference works out of the box.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    SHIM_HOST=0.0.0.0 \
    SHIM_PORT=8009

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .

EXPOSE 8009
# Honor SHIM_PORT at runtime.
CMD ["sh", "-c", "uvicorn server:app --host ${SHIM_HOST} --port ${SHIM_PORT}"]
