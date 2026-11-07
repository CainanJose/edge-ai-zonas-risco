
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

    "prensa": {
        # Zona vermelha: area de operacao em frente a prensa hidraulica
        "vermelha": [(0.29, 0.74), (0.47, 0.74), (0.52, 0.97), (0.29, 0.97)],
        # Zona amarela: faixa de aproximacao ao redor da zona vermelha
        "amarela": [(0.29, 0.66), (0.58, 0.66), (0.68, 1.00), (0.29, 1.00)]
    },

    "obra": {
        # Varias zonas por cor: uma por maquina
        "vermelha": [
            # Raio de giro da escavadeira
            [(0.08, 0.36), (0.36, 0.36), (0.40, 0.72), (0.08, 0.72)],
            # Base e patolas do guindaste
            [(0.60, 0.44), (0.92, 0.44), (0.92, 0.70), (0.60, 0.70)]
        ],
        "amarela": [
            [(0.04, 0.30), (0.40, 0.30), (0.45, 0.92), (0.04, 0.92)],
            [(0.56, 0.38), (0.96, 0.38), (0.96, 0.84), (0.56, 0.84)]
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


def _como_lista_de_poligonos(valor):
    """
    Aceita um unico poligono (lista de tuplas de numeros)
    ou uma lista de poligonos, e retorna sempre uma lista.
    """

    if isinstance(valor[0][0], (int, float)):
        return [valor]

    return valor


def obter_zonas(largura, altura, cenario="padrao"):
    """
    Converte coordenadas normalizadas em pixels
    conforme as dimensoes da imagem.

    Retorna, para cada cor, uma lista de poligonos:
    {"vermelha": [poly, ...], "amarela": [poly, ...]}
    """

    if cenario not in CENARIOS:
        raise ValueError(
            f"Cenario desconhecido: {cenario}"
        )

    zonas = {}

    for nome, valor in CENARIOS[cenario].items():
        zonas[nome] = [
            np.array([
                (
                    int(x * largura),
                    int(y * altura)
                )
                for x, y in pontos
            ], dtype=np.int32)
            for pontos in _como_lista_de_poligonos(valor)
        ]

    return zonas


def _dentro_de_alguma(ponto, poligonos):
    return any(
        cv2.pointPolygonTest(poligono, ponto, False) >= 0
        for poligono in poligonos
    )


def classificar_risco(ponto, zonas):
    """
    Classifica a posicao de uma pessoa conforme as zonas.

    Prioridade:
    1. PERIGO - dentro de qualquer zona vermelha
    2. ATENCAO - dentro de qualquer zona amarela
    3. SEGURO - fora das zonas
    """

    if _dentro_de_alguma(ponto, zonas["vermelha"]):
        return "PERIGO"

    if _dentro_de_alguma(ponto, zonas["amarela"]):
        return "ATENCAO"

    return "SEGURO"
