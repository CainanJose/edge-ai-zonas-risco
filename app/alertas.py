
def gerar_alerta(contagem):
    """
    Gera o alerta conforme a criticidade das deteccoes.

    Prioridade:
    1. CRITICO - pessoa na zona vermelha
    2. ATENCAO - pessoa na zona amarela
    3. NORMAL - nenhuma pessoa em zona de risco
    """

    perigo = contagem.get("PERIGO", 0)
    atencao = contagem.get("ATENCAO", 0)

    if perigo > 0:
        return {
            "ativo": True,
            "nivel": "CRITICO",
            "mensagem": f"{perigo} pessoa(s) na zona vermelha",
            "resposta": "ALERTA_VISUAL_CRITICO"
        }

    if atencao > 0:
        return {
            "ativo": True,
            "nivel": "ATENCAO",
            "mensagem": f"{atencao} pessoa(s) na zona amarela",
            "resposta": "AVISO_VISUAL"
        }

    return {
        "ativo": False,
        "nivel": "NORMAL",
        "mensagem": "Nenhuma pessoa em zona de risco",
        "resposta": "NENHUMA"
    }
