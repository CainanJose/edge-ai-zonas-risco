
import cv2
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.vision import analisar_imagem

app = FastAPI(
    title="Edge AI - Monitoramento de Zonas de Risco",
    description="API de deteccao de pessoas e classificacao de riscos.",
    version="1.0.0"
)


async def carregar_imagem(file: UploadFile):
    conteudo = await file.read()

    if not conteudo:
        raise HTTPException(
            status_code=400,
            detail="Arquivo vazio."
        )

    array = np.frombuffer(conteudo, dtype=np.uint8)
    imagem = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if imagem is None:
        raise HTTPException(
            status_code=400,
            detail="Arquivo nao e uma imagem valida."
        )

    return imagem


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer/json")
async def inferir_json(file: UploadFile = File(...)):
    imagem = await carregar_imagem(file)

    dados, _ = analisar_imagem(imagem)

    return dados


@app.post("/infer/image")
async def inferir_imagem(file: UploadFile = File(...)):
    imagem = await carregar_imagem(file)

    _, anotada = analisar_imagem(imagem)

    sucesso, buffer = cv2.imencode(".png", anotada)

    if not sucesso:
        raise HTTPException(
            status_code=500,
            detail="Erro ao gerar imagem anotada."
        )

    return Response(
        content=buffer.tobytes(),
        media_type="image/png"
    )
