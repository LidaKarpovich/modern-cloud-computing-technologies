# AutoML-as-a-Service — образ для OpenStack VM, Magnum/Kubernetes,
# Yandex Serverless Containers / DataSphere Inference, VK Cloud, Huawei CCE.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MODEL_DIR=/models MAX_TRAIN_WORKERS=1 MAX_TIME_BUDGET=3600

WORKDIR /app
COPY service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY automl_core /app/automl_core
COPY service /app/service
COPY benchmark /app/benchmark

RUN useradd -m -u 1000 automl && mkdir -p /models && chown automl /models
USER automl
VOLUME ["/models"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s CMD python -c \
  "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
