
import time

import cv2
import numpy as np
from ultralytics import YOLO

from app.zones import obter_zonas, classificar_risco
from app.alertas import gerar_alerta


# Modelo carregado apenas uma vez
modelo = YOLO("yolo26n.pt")

# Aquecimento: o primeiro predict e muito mais lento (inicializacao
# interna). Sem isso, a primeira requisicao da API e a media do
# benchmark seriam distorcidas.
modelo.predict(
    source=np.zeros((640, 640, 3), dtype=np.uint8),
    classes=[0],
    device="cpu",
    verbose=False
)


def analisar_imagem(imagem, cenario="padrao"):
    altura, largura = imagem.shape[:2]

    # 1. Configuracao das zonas
    zonas = obter_zonas(largura, altura, cenario)

    # Cada cor pode ter varios poligonos (um por maquina)
    zonas_vermelhas = zonas["vermelha"]
    zonas_amarelas = zonas["amarela"]

    # 2. Desenha as zonas de risco
    sobreposicao = imagem.copy()

    cv2.fillPoly(
        sobreposicao,
        zonas_amarelas,
        (0, 255, 255)
    )

    cv2.fillPoly(
        sobreposicao,
        zonas_vermelhas,
        (0, 0, 255)
    )

    anotada = cv2.addWeighted(
        sobreposicao, 0.15, imagem, 0.85, 0
    )

    # Desenha os contornos
    cv2.polylines(
        anotada,
        zonas_vermelhas,
        True,
        (0, 0, 255),
        3
    )

    cv2.polylines(
        anotada,
        zonas_amarelas,
        True,
        (0, 255, 255),
        3
    )

    # 3. Inferencia do YOLO na CPU
    inicio = time.perf_counter()

    resultado = modelo.predict(
        source=imagem,
        classes=[0],
        conf=0.35,
        device="cpu",
        verbose=False
    )[0]

    tempo_ms = (time.perf_counter() - inicio) * 1000

    # 4. Inicializa resultados
    contagem = {
        "SEGURO": 0,
        "ATENCAO": 0,
        "PERIGO": 0
    }

    deteccoes = []

    cores = {
        "SEGURO": (0, 255, 0),
        "ATENCAO": (0, 255, 255),
        "PERIGO": (0, 0, 255)
    }

    # Texto proporcional a largura, legivel tambem em imagens grandes
    escala = max(0.6, largura / 1200)
    espessura = max(2, int(round(escala * 2)))

    # 5. Processa cada pessoa detectada
    for caixa in resultado.boxes:
        x1, y1, x2, y2 = map(
            int,
            caixa.xyxy[0].tolist()
        )

        confianca = float(caixa.conf[0])

        # Posicao aproximada dos pes
        ponto = (
            (x1 + x2) // 2,
            y2
        )

        # Classificacao geometrica
        risco = classificar_risco(ponto, zonas)

        cor = cores[risco]

        contagem[risco] += 1

        # Registra a deteccao
        deteccoes.append({
            "classe": "person",
            "confianca": round(confianca, 4),
            "bbox": [x1, y1, x2, y2],
            "ponto_referencia": list(ponto),
            "risco": risco
        })

        # Desenha bounding box
        cv2.rectangle(
            anotada,
            (x1, y1),
            (x2, y2),
            cor,
            2
        )

        # Marca o ponto de referencia
        cv2.circle(
            anotada,
            ponto,
            6,
            cor,
            -1
        )

        # Adiciona a legenda.
        # Em pessoas pequenas na imagem (cenas com camera elevada e
        # muitas pessoas), o rotulo completo sobrepoe os vizinhos.
        # Nesse caso, mostra so a confianca com fonte menor; a cor
        # da caixa continua indicando o risco.
        if (y2 - y1) < 0.2 * altura:
            texto = f"{confianca:.0%}"
            escala_rotulo = escala * 0.8
            espessura_rotulo = max(1, int(round(escala_rotulo * 2)))
        else:
            texto = f"pessoa {risco} {confianca:.0%}"
            escala_rotulo = escala
            espessura_rotulo = espessura

        tamanho, _ = cv2.getTextSize(
            texto,
            cv2.FONT_HERSHEY_SIMPLEX,
            escala_rotulo,
            espessura_rotulo
        )

        # Mantem o texto dentro dos limites da imagem
        x_texto = max(
            0,
            min(x1, largura - tamanho[0])
        )

        y_texto = max(tamanho[1] + 5, y1 - 10)

        cv2.putText(
            anotada,
            texto,
            (x_texto, y_texto),
            cv2.FONT_HERSHEY_SIMPLEX,
            escala_rotulo,
            cor,
            espessura_rotulo
        )

    # 6. Gera o alerta automaticamente
    alerta = gerar_alerta(contagem)

    # 7. Desenha um aviso visual quando ha risco
    if alerta["ativo"]:
        cor_alerta = (
            (0, 0, 255)
            if alerta["nivel"] == "CRITICO"
            else (0, 255, 255)
        )

        texto_alerta = (
            f"{alerta['nivel']}: {alerta['mensagem']}"
        )

        (largura_txt, altura_txt), base = cv2.getTextSize(
            texto_alerta,
            cv2.FONT_HERSHEY_SIMPLEX,
            escala,
            espessura
        )

        # Faixa escura no topo, com altura proporcional ao texto
        margem = int(altura_txt * 0.6)
        altura_barra = min(
            altura,
            altura_txt + base + 2 * margem
        )

        fundo = anotada.copy()

        cv2.rectangle(
            fundo,
            (0, 0),
            (largura - 1, altura_barra - 1),
            (20, 20, 20),
            -1
        )

        anotada = cv2.addWeighted(
            fundo, 0.75, anotada, 0.25, 0
        )

        # Indicador colorido
        largura_indicador = max(8, int(8 * escala))

        cv2.rectangle(
            anotada,
            (0, 0),
            (largura_indicador, altura_barra - 1),
            cor_alerta,
            -1
        )

        cv2.putText(
            anotada,
            texto_alerta,
            (
                largura_indicador + 10,
                min(margem + altura_txt, altura - 5)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            escala,
            cor_alerta,
            espessura
        )

    # 8. Monta o resultado final
    dados = {
        "cenario": cenario,
        "modelo": "yolo26n.pt",
        "largura": largura,
        "altura": altura,
        "total_pessoas": len(deteccoes),
        "contagem": contagem,
        "tempo_inferencia_ms": round(tempo_ms, 2),
        "alerta": alerta,
        "deteccoes": deteccoes
    }

    return dados, anotada
