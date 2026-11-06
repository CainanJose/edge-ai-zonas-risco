
from pathlib import Path

import cv2

origem = Path("examples/imagens/test.jpg")
destino = Path("examples/videos/teste_estatico.mp4")

destino.parent.mkdir(parents=True, exist_ok=True)

imagem = cv2.imread(str(origem))

if imagem is None:
    raise FileNotFoundError(
        f"Imagem nao encontrada: {origem}"
    )

altura, largura = imagem.shape[:2]

nova_largura = 640
nova_altura = int(altura * nova_largura / largura)

imagem = cv2.resize(
    imagem,
    (nova_largura, nova_altura)
)

codec = cv2.VideoWriter_fourcc(*"mp4v")

video = cv2.VideoWriter(
    str(destino),
    codec,
    10,
    (nova_largura, nova_altura)
)

if not video.isOpened():
    raise RuntimeError("Erro ao criar video de teste.")

try:
    for _ in range(30):
        video.write(imagem)
finally:
    video.release()

print(f"Video de teste criado: {destino}")
