
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


def test_prensa_zona_vermelha():
    zonas = obter_zonas(1000, 1000, "prensa")
    assert classificar_risco((379, 853), zonas) == "PERIGO"


def test_prensa_zona_segura():
    zonas = obter_zonas(1000, 1000, "prensa")
    assert classificar_risco((905, 587), zonas) == "SEGURO"


def test_obra_zona_vermelha():
    zonas = obter_zonas(1000, 1000, "obra")
    assert classificar_risco((343, 670), zonas) == "PERIGO"


def test_obra_zona_amarela():
    zonas = obter_zonas(1000, 1000, "obra")
    assert classificar_risco((216, 882), zonas) == "ATENCAO"


def test_obra_zona_amarela_guindaste():
    zonas = obter_zonas(1000, 1000, "obra")
    assert classificar_risco((778, 801), zonas) == "ATENCAO"


def test_obra_zona_vermelha_guindaste():
    zonas = obter_zonas(1000, 1000, "obra")
    assert classificar_risco((760, 560), zonas) == "PERIGO"


def test_obra_zona_segura():
    zonas = obter_zonas(1000, 1000, "obra")
    assert classificar_risco((450, 410), zonas) == "SEGURO"
