
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Carrega a imagem de teste
imagem = cv2.imread("bus.jpg")

if imagem is None:
    raise FileNotFoundError("Imagem bus.jpg nao encontrada.")

altura, largura = imagem.shape[:2]

# Zonas definidas proporcionalmente ao tamanho da imagem
zona_vermelha = np.array([
    (0, int(altura * 0.65)),
    (int(largura * 0.25), int(altura * 0.65)),
    (int(largura * 0.25), altura),
    (0, altura)
], dtype=np.int32)

zona_amarela = np.array([
    (int(largura * 0.25), int(altura * 0.65)),
    (int(largura * 0.48), int(altura * 0.65)),
    (int(largura * 0.48), altura),
    (int(largura * 0.25), altura)
], dtype=np.int32)

# Desenha as zonas com transparência
sobreposicao = imagem.copy()

cv2.fillPoly(sobreposicao, [zona_vermelha], (0, 0, 255))
cv2.fillPoly(sobreposicao, [zona_amarela], (0, 255, 255))

imagem = cv2.addWeighted(
    sobreposicao, 0.25, imagem, 0.75, 0
)

# Executa a detecção de pessoas
modelo = YOLO("yolo26n.pt")

resultado = modelo.predict(
    source="bus.jpg",
    classes=[0],
    conf=0.35,
    device="cpu",
    verbose=False
)[0]

contagem = {"SEGURO": 0, "ATENCAO": 0, "PERIGO": 0}

for caixa in resultado.boxes:
    x1, y1, x2, y2 = map(int, caixa.xyxy[0].tolist())
    confianca = float(caixa.conf[0])

    # Ponto inferior central da pessoa
    ponto = ((x1 + x2) // 2, y2)

    # Verifica em qual zona a pessoa esta
    if cv2.pointPolygonTest(zona_vermelha, ponto, False) >= 0:
        risco = "PERIGO"
        cor = (0, 0, 255)

    elif cv2.pointPolygonTest(zona_amarela, ponto, False) >= 0:
        risco = "ATENCAO"
        cor = (0, 255, 255)

    else:
        risco = "SEGURO"
        cor = (0, 255, 0)

    contagem[risco] += 1

    # Desenha a caixa da pessoa
    cv2.rectangle(imagem, (x1, y1), (x2, y2), cor, 2)

    # Identifica risco e confianca
    texto = f"{risco} {confianca:.0%}"

    cv2.putText(
        imagem,
        texto,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        cor,
        2
    )

    # Marca a posicao dos pes
    cv2.circle(imagem, ponto, 6, cor, -1)

    print(f"Pessoa detectada: {risco} | Confianca: {confianca:.2%}")

# Salva a imagem
Path("outputs").mkdir(exist_ok=True)

cv2.imwrite("outputs/deteccao_zonas.jpg", imagem)

print("\nResumo:")
print(contagem)
print("Imagem salva em outputs/deteccao_zonas.jpg")
