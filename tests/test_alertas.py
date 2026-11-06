
from app.alertas import gerar_alerta


def test_sem_alerta():
    contagem = {
        "SEGURO": 2,
        "ATENCAO": 0,
        "PERIGO": 0
    }

    alerta = gerar_alerta(contagem)

    assert alerta["ativo"] is False
    assert alerta["nivel"] == "NORMAL"


def test_alerta_atencao():
    contagem = {
        "SEGURO": 0,
        "ATENCAO": 2,
        "PERIGO": 0
    }

    alerta = gerar_alerta(contagem)

    assert alerta["ativo"] is True
    assert alerta["nivel"] == "ATENCAO"
    assert alerta["resposta"] == "AVISO_VISUAL"


def test_alerta_critico():
    contagem = {
        "SEGURO": 0,
        "ATENCAO": 0,
        "PERIGO": 1
    }

    alerta = gerar_alerta(contagem)

    assert alerta["ativo"] is True
    assert alerta["nivel"] == "CRITICO"
    assert alerta["resposta"] == "ALERTA_VISUAL_CRITICO"


def test_prioridade_alerta_critico():
    contagem = {
        "SEGURO": 1,
        "ATENCAO": 2,
        "PERIGO": 1
    }

    alerta = gerar_alerta(contagem)

    assert alerta["nivel"] == "CRITICO"
