
from app.zones import obter_zonas, classificar_risco


def test_pessoa_zona_vermelha():
    zonas = obter_zonas(1000, 1000, "deposito")
    assert classificar_risco((500, 700), zonas) == "PERIGO"


def test_pessoa_zona_amarela():
    zonas = obter_zonas(1000, 1000, "deposito")
    assert classificar_risco((700, 700), zonas) == "ATENCAO"


def test_pessoa_zona_segura():
    zonas = obter_zonas(1000, 1000, "deposito")
    assert classificar_risco((100, 100), zonas) == "SEGURO"


def test_prioridade_zona_vermelha():
    zonas = obter_zonas(1000, 1000, "deposito")
    assert classificar_risco((570, 700), zonas) == "PERIGO"
