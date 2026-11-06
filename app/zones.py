
import cv2
import numpy as np

# Coordenadas normalizadas entre 0 e 1.
# Cada configuracao representa uma camera fixa.

CENARIOS = {
    "padrao": {
        "vermelha": [
            (0.00, 0.65),
            (0.25, 0.65),
            (0.25, 1.00),
            (0.00, 1.00)
        ],
        "amarela": [
            (0.25, 0.65),
            (0.48, 0.65),
            (0.48, 1.00),
            (0.25, 1.00)
        ]
    },

    "deposito": {
        # Zona vermelha: piso proximo a empilhadeira
        "vermelha": [
            (0.43, 0.64),
            (0.55, 0.64),
            (0.63, 0.84),
            (0.31, 0.84)
        ],

        # Zona amarela: faixa de aproximacao dos pedestres
        "amarela": [
            (0.56, 0.64),
            (0.76, 0.64),
            (0.88, 0.84),
            (0.58, 0.84)
        ]
    },

    "corredor": {
        # Regiao de operacao da empilhadeira
        "vermelha": [
            (0.46, 0.50),
            (0.83, 0.50),
            (0.90, 1.00),
            (0.40, 1.00)
        ],
        "amarela": [
            (0.27, 0.47),
            (0.90, 0.47),
            (0.98, 1.00),
            (0.10, 1.00)
        ]
    },

    "webcam": {
        # Terco esquerdo da tela: PERIGO
        "vermelha": [
            (0.00, 0.00),
            (0.3333, 0.00),
            (0.3333, 1.00),
            (0.00, 1.00)
        ],

        # Terco central da tela: ATENCAO
        "amarela": [
            (0.3333, 0.00),
            (0.6667, 0.00),
            (0.6667, 1.00),
            (0.3333, 1.00)
        ]

        # Terco direito: SEGURO
        # Nao necessita de poligono.
    }
}


def obter_zonas(largura, altura, cenario="padrao"):
    """
    Converte coordenadas normalizadas em pixels
    conforme as dimensoes da imagem.
    """

    if cenario not in CENARIOS:
        raise ValueError(
            f"Cenario desconhecido: {cenario}"
        )

    zonas = {}

    for nome, pontos in CENARIOS[cenario].items():
        zonas[nome] = np.array([
            (
                int(x * largura),
                int(y * altura)
            )
            for x, y in pontos
        ], dtype=np.int32)

    return zonas


def classificar_risco(ponto, zonas):
    """
    Classifica a posicao de uma pessoa conforme as zonas.

    Prioridade:
    1. PERIGO - zona vermelha
    2. ATENCAO - zona amarela
    3. SEGURO - fora das zonas
    """

    if cv2.pointPolygonTest(
        zonas["vermelha"],
        ponto,
        False
    ) >= 0:
        return "PERIGO"

    if cv2.pointPolygonTest(
        zonas["amarela"],
        ponto,
        False
    ) >= 0:
        return "ATENCAO"

    return "SEGURO"
