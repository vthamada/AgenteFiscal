import pytest
from banco_de_dados import BancoDeDados

@pytest.fixture
def db(tmp_path):
    db = BancoDeDados(db_path=tmp_path / "teste.sqlite")
    yield db
    db.close()

def test_inserir_documento(db):
    doc_id = db.inserir_documento(
        nome_arquivo="teste.xml",
        tipo="NFe",
        origem="unittest",
        hash="hash123",
        status="processando",
        emitente_cnpj="12345678000195",
        destinatario_cnpj="98765432000100",
        valor_total=100.0
    )
    assert doc_id > 0
    doc = db.get_documento(doc_id)
    assert doc["nome_arquivo"] == "teste.xml"

def test_inserir_item_e_imposto(db):
    doc_id = db.inserir_documento(nome_arquivo="nfe.xml", hash="abc", tipo="NFe", origem="tst")
    item_id = db.inserir_item(documento_id=doc_id, descricao="Produto A", valor_total=10.0)
    imp_id = db.inserir_imposto(item_id=item_id, tipo_imposto="ICMS", cst="00", valor=1.8)
    assert item_id > 0
    assert imp_id > 0
