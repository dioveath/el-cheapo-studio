FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY server/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY server/ ./
COPY examples/ /app/examples/
CMD ["uvicorn", "studio.api:app", "--host", "0.0.0.0", "--port", "8000"]
