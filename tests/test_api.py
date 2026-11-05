
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def criar_imagem_teste():
    """Gera uma imagem simples sem depender de arquivos externos."""
    imagem = np.full((320, 320, 3), 255, dtype=np.uint8)
    sucesso, buffer = cv2.imencode(".jpg", imagem)

    assert sucesso
    return buffer.tobytes()


def test_health():
    resposta = client.get("/health")

    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_inferencia_json():
    imagem = criar_imagem_teste()

    resposta = client.post(
        "/infer/json",
        files={"file": ("teste.jpg", imagem, "image/jpeg")}
    )

    assert resposta.status_code == 200

    dados = resposta.json()

    assert dados["modelo"] == "yolo26n.pt"
    assert "deteccoes" in dados
    assert "contagem" in dados
    assert "tempo_inferencia_ms" in dados
    assert dados["total_pessoas"] == len(dados["deteccoes"])
    assert sum(dados["contagem"].values()) == dados["total_pessoas"]


def test_inferencia_imagem():
    imagem = criar_imagem_teste()

    resposta = client.post(
        "/infer/image",
        files={"file": ("teste.jpg", imagem, "image/jpeg")}
    )

    assert resposta.status_code == 200
    assert resposta.headers["content-type"] == "image/png"
    assert resposta.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_imagem_invalida():
    resposta = client.post(
        "/infer/json",
        files={"file": ("invalido.jpg", b"arquivo-invalido", "image/jpeg")}
    )

    assert resposta.status_code == 400
