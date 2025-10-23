import pytest
from seguranca import Cofre, gerar_chave, mascarar_cnpj, mascarar_documento_fiscal

def test_criptografia_basica():
    key = gerar_chave()
    cofre = Cofre(key)
    texto = "dados confidenciais"
    cript = cofre.encrypt_text(texto)
    decript = cofre.decrypt_text(cript)
    assert texto == decript

def test_mascaramento_cnpj():
    cnpj = "12345678000195"
    masked = mascarar_cnpj(cnpj)
    assert "***" in masked
    assert masked.startswith("12.")

def test_mascaramento_documento():
    # CNPJ válido → deve aplicar máscara com ***
    masked_cnpj = mascarar_documento_fiscal("12345678000195")
    assert masked_cnpj.startswith("12.") and "***" in masked_cnpj and masked_cnpj.endswith("-95")

    # CPF válido → deve aplicar máscara com ***
    masked_cpf = mascarar_documento_fiscal("12345678909")
    assert masked_cpf.startswith("123.") and "***" in masked_cpf and masked_cpf.endswith("-09")

    # Caso vazio → deve retornar vazio
    assert mascarar_documento_fiscal("") == ""

    # Caso não numérico → retorna parcialmente oculto
    assert mascarar_documento_fiscal("ISENTO").endswith("...")