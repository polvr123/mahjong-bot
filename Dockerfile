# Riichi mahjong RL demo. Build: docker build -t mahjong-bot .   Run: docker run -p 7860:7860 mahjong-bot
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1

WORKDIR /app

# CPU-only torch first (the default Linux wheel bundles CUDA and is ~10x larger), then everything else.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Application code + the trained checkpoint (mahjong_rl/serving/models/model.zip)
COPY mahjong_rl ./mahjong_rl

# Hugging Face Spaces expects 7860; most other hosts inject $PORT.
ENV PORT=7860
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request as u; u.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"7860\")}/health')" || exit 1

CMD ["sh", "-c", "python -m uvicorn mahjong_rl.serving.api:app --host 0.0.0.0 --port ${PORT}"]
