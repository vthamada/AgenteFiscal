import pytest
from banco_de_dados import BancoDeDados
from agentes import MetricsAgent

@pytest.fixture
def db(tmp_path):
    db = BancoDeDados(db_path=tmp_path / "teste.sqlite")
    yield db
    db.close()

def test_registrar_metrica(db):
    agente = MetricsAgent()
    agente.registrar_metrica(
        db=db,
        tipo_documento="NFe",
        status="processado",
        confianca_media=0.95,
        tempo_medio=1.2
    )
    metricas = db.query_table("metricas")
    assert not metricas.empty
