# testes/test_fluxo_integrado.py
"""
Teste de integração completo do sistema fiscal.
Fluxo: Documento → Itens → Impostos → Validação → Métrica
"""

import pytest
from pathlib import Path
from banco_de_dados import BancoDeDados
from seguranca import Cofre, gerar_chave
from validacao import ValidadorFiscal
from agentes import MetricsAgent


@pytest.fixture(scope="module")
def db_tmp(tmp_path_factory):
    """Cria um banco SQLite temporário para os testes."""
    db_path = tmp_path_factory.mktemp("data") / "teste_integrado.sqlite"
    db = BancoDeDados(db_path=db_path)
    yield db
    db.close()


def test_fluxo_integrado_documento_validado(db_tmp):
    """
    Cenário:
    1. Cria documento fiscal simulado
    2. Insere itens e impostos
    3. Valida o documento via ValidadorFiscal
    4. Gera métrica via MetricsAgent
    5. Verifica se todas as etapas atualizaram o banco corretamente
    """

    # --- Etapa 1: Criação do Cofre e componentes ---
    key = gerar_chave()
    cofre = Cofre(key=key)
    validador = ValidadorFiscal(cofre=cofre)
    metrics = MetricsAgent()

    # --- Etapa 2: Inserção do Documento ---
    doc_id = db_tmp.inserir_documento(
        nome_arquivo="nota_fiscal_001.xml",
        tipo="NFe",
        origem="upload",
        hash="abc123",
        status="processando",
        data_upload="2025-10-22T00:00:00Z",
        emitente_cnpj=cofre.encrypt_text("29481253000106"),  # CNPJ válido
        emitente_nome="Empresa Exemplo LTDA",
        destinatario_cnpj=cofre.encrypt_text("10955024000158"),  # CNPJ válido
        destinatario_nome="Cliente XYZ",
        municipio="São Paulo",  # Adiciona município válido
        valor_total=150.00,
        caminho_arquivo=str(Path("data/uploads/nota_fiscal_001.xml")),
    )

    assert doc_id > 0

    # --- Etapa 3: Inserção de Itens ---
    item_id_1 = db_tmp.inserir_item(
        documento_id=doc_id,
        descricao="Produto A",
        ncm="27101932",
        cfop="5101",
        quantidade=2,
        unidade="UN",
        valor_unitario=50.00,
        valor_total=100.00,
        codigo_produto="A001"
    )

    item_id_2 = db_tmp.inserir_item(
        documento_id=doc_id,
        descricao="Produto B",
        ncm="22030000",
        cfop="5102",
        quantidade=1,
        unidade="CX",
        valor_unitario=50.00,
        valor_total=50.00,
        codigo_produto="B002"
    )

    assert item_id_1 > 0 and item_id_2 > 0

    # --- Etapa 4: Inserção de Impostos ---
    imp_1 = db_tmp.inserir_imposto(
        item_id=item_id_1,
        tipo_imposto="ICMS",
        cst="00",
        base_calculo=100.00,
        aliquota=18.00,
        valor=18.00
    )

    imp_2 = db_tmp.inserir_imposto(
        item_id=item_id_2,
        tipo_imposto="PIS",
        cst="01",
        base_calculo=50.00,
        aliquota=1.65,
        valor=0.83
    )

    assert imp_1 > 0 and imp_2 > 0

    # --- Etapa 5: Validação Fiscal ---
    validador.validar_documento(doc_id=doc_id, db=db_tmp)

    doc = db_tmp.get_documento(doc_id)
    assert doc is not None
    assert doc["status"] == "processado"  # Agora deve ser validado com sucesso
    assert doc["motivo_rejeicao"] is None or doc["motivo_rejeicao"] == ""

    # --- Etapa 6: Registro de Métrica ---
    metrics.registrar_metrica(
        db=db_tmp,
        tipo_documento="NFe",
        status=doc["status"],
        confianca_media=0.95,
        tempo_medio=0.7
    )

    df_metrics = db_tmp.query_table("metricas")
    assert not df_metrics.empty
    assert "acuracia_media" in df_metrics.columns
    assert df_metrics.iloc[0]["tipo_documento"] == "NFe"

    # --- Etapa 7: Logs e Integridade ---
    df_logs = db_tmp.query_table("logs")
    assert not df_logs.empty
    assert any("validacao_ok" in evt for evt in df_logs["evento"].tolist())


def test_fluxo_integrado_documento_invalido(db_tmp):
    """
    Testa cenário de documento inconsistente (total errado).
    """
    key = gerar_chave()
    cofre = Cofre(key=key)
    validador = ValidadorFiscal(cofre=cofre)

    # Documento com total inconsistente
    doc_id = db_tmp.inserir_documento(
        nome_arquivo="nota_fiscal_erro.xml",
        tipo="NFe",
        origem="upload",
        hash="hash-erro",
        status="processando",
        data_upload="2025-10-22T00:00:00Z",
        emitente_cnpj=cofre.encrypt_text("29481253000106"), 
        emitente_nome="Fornecedor Inválido SA",
        destinatario_cnpj=cofre.encrypt_text("10955024000158"), 
        destinatario_nome="Cliente XYZ",
        municipio="São Paulo",
        valor_total=200.00,  # Deve causar diferença
        caminho_arquivo=str(Path("data/uploads/nota_fiscal_erro.xml")),
    )

    item_id = db_tmp.inserir_item(
        documento_id=doc_id,
        descricao="Produto C",
        ncm="27101932",
        cfop="5101",
        quantidade=1,
        unidade="UN",
        valor_unitario=100.00,
        valor_total=100.00,
        codigo_produto="C003"
    )

    assert item_id > 0

    # Validação deve marcar como revisão
    validador.validar_documento(doc_id=doc_id, db=db_tmp)
    doc = db_tmp.get_documento(doc_id)

    assert doc["status"] == "revisao_pendente"
    assert "totais" in (doc["motivo_rejeicao"] or "").lower()
