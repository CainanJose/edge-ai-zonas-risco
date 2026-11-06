
from pathlib import Path

import cv2
from ultralytics import YOLO

# Diretorios
PASTA_IMAGENS = Path("examples/imagens")
PASTA_SAIDA = Path("outputs/industriais")

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

# Carrega o modelo apenas uma vez
modelo = YOLO("yolo26n.pt")

# Localiza todas as imagens JPG
imagens = sorted(PASTA_IMAGENS.glob("*.jpg"))

if not imagens:
    raise FileNotFoundError(
        f"Nenhuma imagem encontrada em {PASTA_IMAGENS}"
    )

print(f"\nImagens encontradas: {len(imagens)}\n")

for caminho in imagens:
    print(f"Processando: {caminho.name}")

    # Inferencia de pessoas
    resultado = modelo.predict(
        source=str(caminho),
        classes=[0],
        conf=0.35,
        device="cpu",
        verbose=False
    )[0]

    total = len(resultado.boxes)

    # Imagem com bounding boxes do YOLO
    imagem_anotada = resultado.plot()

    # Salva o resultado
    destino = PASTA_SAIDA / f"detectado_{caminho.name}"

    if not cv2.imwrite(str(destino), imagem_anotada):
        raise RuntimeError(f"Erro ao salvar {destino}")

    print(f"Pessoas detectadas: {total}")

    for indice, caixa in enumerate(resultado.boxes, 1):
        confianca = float(caixa.conf[0])
        print(f"  Pessoa {indice}: {confianca:.2%}")

    print(f"Resultado salvo: {destino}\n")

print("Todos os testes foram finalizados!")
