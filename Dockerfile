FROM python:3.12.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

# The default PyPI torch wheel for Linux pulls in CUDA libraries; the CPU build keeps the image small.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so the first video load does not wait for a download.
# Only config.py is copied here, so code changes elsewhere do not invalidate this layer.
COPY askthevideo/__init__.py askthevideo/config.py ./askthevideo/
RUN python -c "from sentence_transformers import SentenceTransformer; from askthevideo.config import EMBEDDING_MODEL; SentenceTransformer(EMBEDDING_MODEL, device='cpu')"

COPY .streamlit ./.streamlit
COPY app.py .
COPY askthevideo ./askthevideo

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
