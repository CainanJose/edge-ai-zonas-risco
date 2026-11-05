
from pathlib import Path

import cv2
from ultralytics import YOLO

# Carrega o modelo pré-treinado.
modelo = YOLO("yolo26n.pt")

# Executa a detecção apenas para a classe pessoa (COCO: 0).
resultado = modelo.predict(
    source="https://ultralytics.com/images/bus.jpg",
    classes=[0],
    conf=0.35,
    device="cpu",
    verbose=False
)[0]

# Salva a imagem com as detecções.
Path("outputs").mkdir(exist_ok=True)

imagem_anotada = resultado.plot()
cv2.imwrite("outputs/deteccao_pessoas.jpg", imagem_anotada)

# Exibe as detecções encontradas.
for indice, caixa in enumerate(resultado.boxes, start=1):
    confianca = float(caixa.conf[0])
    coordenadas = caixa.xyxy[0].tolist()

    print(
        f"Pessoa {indice}: "
        f"confianca={confianca:.2%}, "
        f"coordenadas={coordenadas}"
    )

print("Imagem salva em outputs/deteccao_pessoas.jpg")
