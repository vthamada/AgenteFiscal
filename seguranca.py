# seguranca.py

from __future__ import annotations
from pathlib import Path
from typing import Optional
import hashlib
import os
import base64
import re # Importado para as funções de mascaramento

# Criptografia é opcional — só habilita se a lib estiver disponível
try:
    from cryptography.fernet import Fernet  # type: ignore
    CRYPTO_OK = True
except Exception:
    CRYPTO_OK = False


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))

# --- Funções de Mascaramento (Novas) ---

def mascarar_cpf(cpf_limpo: str) -> str:
    """
    Aplica máscara a um CPF (ex: 123.***.***-99).
    Espera receber uma string com exatamente 11 dígitos.
    """
    if not cpf_limpo or len(cpf_limpo) != 11 or not cpf_limpo.isdigit():
        return "CPF Inválido" # Retorna indicação de erro se não forem 11 dígitos
    return f"{cpf_limpo[:3]}.***.***-{cpf_limpo[9:]}"

def mascarar_cnpj(cnpj_limpo: str) -> str:
    """
    Aplica máscara a um CNPJ (ex: 12.***.***/0001-99).
    Espera receber uma string com exatamente 14 dígitos.
    """
    if not cnpj_limpo or len(cnpj_limpo) != 14 or not cnpj_limpo.isdigit():
        return "CNPJ Inválido" # Retorna indicação de erro se não forem 14 dígitos
    return f"{cnpj_limpo[:2]}.***.***/{cnpj_limpo[8:12]}-{cnpj_limpo[12:]}"

def mascarar_documento_fiscal(doc_str: Optional[str]) -> str:
    """
    Aplica máscara de CPF ou CNPJ detectando pelo tamanho.
    Recebe o valor limpo (só dígitos) ou formatado.
    """
    if not doc_str:
        return ""
    doc_limpo = re.sub(r"\D", "", doc_str) # Limpa formatação
    if len(doc_limpo) == 11:
        return mascarar_cpf(doc_limpo)
    if len(doc_limpo) == 14:
        return mascarar_cnpj(doc_limpo)
    # Se não for CPF/CNPJ (ex: "ISENTO" ou tamanho errado), retorna truncado/oculto
    return f"{doc_str[:4]}..." if len(doc_str) > 4 else doc_str

# --- Funções de Criptografia ---

def gerar_chave() -> Optional[bytes]:
    """
    Gera uma chave simétrica (Fernet). Retorna None se cryptography não estiver disponível.
    """
    if not CRYPTO_OK:
        return None
    return Fernet.generate_key()


def carregar_chave_do_env(var_name: str = "APP_SECRET_KEY") -> Optional[bytes]:
    """
    Lê uma chave base64 do ambiente (ex.: export APP_SECRET_KEY="<base64>").
    Retorna None se não houver chave ou se cryptography estiver indisponível.
    """
    if not CRYPTO_OK:
        return None
    val = os.getenv(var_name) or ""
    val = val.strip()
    if not val:
        return None
    try:
        # Valida se é uma chave Fernet válida (base64 urlsafe 32 bytes)
        Fernet(val.encode("utf-8"))
        return val.encode("utf-8")
    except Exception:
        # Como fallback, aceita 32 bytes em base64 padrão e converte
        try:
            raw = base64.b64encode(base64.b64decode(val))
            Fernet(raw)
            return raw
        except Exception:
            return None


class Cofre:
    """
    Wrapper simples para cifrar/decifrar conteúdos com Fernet (opcional).
    Se cryptography não estiver disponível, métodos retornam o conteúdo puro.
    """
    def __init__(self, key: Optional[bytes] = None) -> None:
        self.available = CRYPTO_OK and (key is not None)
        self._fernet = Fernet(key) if self.available and key else None

    def encrypt_bytes(self, data: bytes) -> bytes:
        if not self.available or not self._fernet:
            return data
        return self._fernet.encrypt(data)

    def decrypt_bytes(self, token: bytes) -> bytes:
        if not self.available or not self._fernet:
            return token
        # Adiciona tratamento de erro para token inválido
        try:
            return self._fernet.decrypt(token)
        except Exception as e:
            logging.warning(f"Falha ao descriptografar token: {e}. Retornando token original.")
            # Retorna o token original (ilegível) em caso de falha de descriptografia
            return token

    def encrypt_text(self, text: str) -> str:
        """Criptografa texto (str) e retorna base64 (str)."""
        if not self.available or not self._fernet:
            return text # Retorna texto original se criptografia inativa
        if not text:
            return text # Não tenta criptografar string vazia
        try:
            out = self.encrypt_bytes(text.encode("utf-8"))
            return base64.b64encode(out).decode("utf-8")
        except Exception as e:
            logging.error(f"Erro ao criptografar texto: {e}")
            return text # Retorna original em caso de erro

    def decrypt_text(self, text: str) -> str:
        """Descriptografa texto (base64 str) e retorna (str)."""
        if not self.available or not self._fernet:
            return text # Retorna texto original se criptografia inativa
        if not text:
            return text # Não tenta descriptografar string vazia
        try:
            raw = base64.b64decode(text.encode("utf-8"))
            plain = self.decrypt_bytes(raw)
            return plain.decode("utf-8")
        except (base64.binascii.Error, TypeError):
             # Se o texto não for base64 válido (ex: já é texto puro), retorna o original
             return text
        except Exception as e:
            # Outros erros de descriptografia (ex: Chave errada)
            logging.warning(f"Falha ao descriptografar texto '{text[:10]}...': {e}")
            return text # Retorna o texto original (criptografado/ilegível)