FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# CPU-only PyTorch first (the default PyPI wheel pulls multi-GB CUDA libraries).
COPY requirements.txt .
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY app ./app
COPY api ./api
COPY ui ./ui
COPY scripts ./scripts
COPY evaluation ./evaluation
COPY .streamlit ./.streamlit

ENV PYTHONPATH=/app
EXPOSE 8000 8501

# Default: API. docker-compose overrides the command for the UI and one-off jobs.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
