import pytest
from banco_de_dados import BancoDeDados
from validacao import ValidadorFiscal

@pytest.fixture
def db(tmp_path):
    db = BancoDeDados(db_path=tmp_path / "teste.sqlite")
    yield db
    db.close()

def test_validacao_cnpj_invalido(db):
    doc_id = db.inserir_documento(
        nome_arquivo="nota_invalida.xml",
        tipo="NFe",
        hash="h1",
        emitente_cnpj="11111111111111",  # Inválido
        destinatario_cnpj="12345678000195",
        valor_total=50.0,
        status="processando"
    )
    val = ValidadorFiscal()
    val.validar_documento(doc_id=doc_id, db=db, force_revalidation=True)
    doc = db.get_documento(doc_id)
    assert doc["status"] == "revisao_pendente"

def test_validacao_total_errado(db):
    doc_id = db.inserir_documento(
        nome_arquivo="nota_total.xml",
        tipo="NFe",
        hash="h2",
        emitente_cnpj="12345678000195",
        destinatario_cnpj="98765432000100",
        valor_total=100.0,
        status="processando"
    )
    item_id = db.inserir_item(documento_id=doc_id, descricao="Produto A", valor_total=80.0)
    val = ValidadorFiscal()
    val.validar_documento(doc_id=doc_id, db=db, force_revalidation=True)
    doc = db.get_documento(doc_id)
    assert doc["status"] == "revisao_pendente"
