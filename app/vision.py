
import time

import cv2
import numpy as np
from ultralytics import YOLO

modelo = YOLO("yolo26n.pt")


def analisar_imagem(imagem):
    altura, largura = imagem.shape[:2]

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

    # Desenha as zonas na imagem
    sobreposicao = imagem.copy()
    cv2.fillPoly(sobreposicao, [zona_vermelha], (0, 0, 255))
    cv2.fillPoly(sobreposicao, [zona_amarela], (0, 255, 255))

    anotada = cv2.addWeighted(
        sobreposicao, 0.25, imagem, 0.75, 0
    )

    # Inferencia local na CPU
    inicio = time.perf_counter()

    resultado = modelo.predict(
        source=imagem,
        classes=[0],
        conf=0.35,
        device="cpu",
        verbose=False
    )[0]

    tempo_ms = (time.perf_counter() - inicio) * 1000

    contagem = {"SEGURO": 0, "ATENCAO": 0, "PERIGO": 0}
    deteccoes = []

    for caixa in resultado.boxes:
        x1, y1, x2, y2 = map(int, caixa.xyxy[0].tolist())
        confianca = float(caixa.conf[0])

        ponto = ((x1 + x2) // 2, y2)

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

        deteccoes.append({
            "classe": "person",
            "confianca": round(confianca, 4),
            "bbox": [x1, y1, x2, y2],
            "ponto_referencia": list(ponto),
            "risco": risco
        })

        cv2.rectangle(anotada, (x1, y1), (x2, y2), cor, 2)
        cv2.circle(anotada, ponto, 6, cor, -1)

        cv2.putText(
            anotada,
            f"{risco} {confianca:.0%}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            cor,
            2
        )

    dados = {
        "modelo": "yolo26n.pt",
        "largura": largura,
        "altura": altura,
        "total_pessoas": len(deteccoes),
        "contagem": contagem,
        "tempo_inferencia_ms": round(tempo_ms, 2),
        "deteccoes": deteccoes
    }

    return dados, anotada
