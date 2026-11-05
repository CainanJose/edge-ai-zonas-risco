
FROM python:3.11-slim-bookworm

ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

# Biblioteca de execucao para calculos numericos
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Instala PyTorch apropriado para a arquitetura
RUN python -m pip install --no-cache-dir --upgrade pip && \
    if [ "$TARGETARCH" = "amd64" ]; then \
        python -m pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu; \
    else \
        python -m pip install --no-cache-dir \
        torch torchvision; \
    fi

# Dependencias da aplicacao
COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt


# Remove variantes conflitantes do OpenCV
RUN python -m pip uninstall -y \
        opencv-python \
        opencv-contrib-python \
        opencv-python-headless \
        opencv-contrib-python-headless && \
    python -m pip install --no-cache-dir opencv-python-headless && \
    python -c "import cv2; print('OpenCV OK:', cv2.__version__)"


# Codigo e pesos do modelo
COPY app/ ./app/
COPY yolo26n.pt ./yolo26n.pt

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
