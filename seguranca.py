# seguranca.py

from __future__ import annotations
from pathlib import Path
from typing import Optional
import hashlib
import os
import base64
import re
import logging

# --- Configuração do logger ---
log = logging.getLogger("projeto_fiscal.seguranca")
if not log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)

# --- Criptografia opcional ---
try:
    from cryptography.fernet import Fernet  # type: ignore
    CRYPTO_OK = True
except Exception:
    CRYPTO_OK = False
    Fernet = None
    log.warning("Biblioteca 'cryptography' não encontrada. Criptografia desativada.")

# --- Funções de hash (sem alteração) ---
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

# --- Funções de mascaramento ---
def mascarar_cpf(cpf_limpo: str) -> str:
    """Aplica máscara a um CPF (ex: 123.***.***-99)."""
    if not cpf_limpo or len(cpf_limpo) != 11 or not cpf_limpo.isdigit():
        return "CPF Inválido"
    return f"{cpf_limpo[:3]}.***.***-{cpf_limpo[9:]}"

def mascarar_cnpj(cnpj_limpo: str) -> str:
    """Aplica máscara a um CNPJ (ex: 12.***.***/0001-99)."""
    if not cnpj_limpo or len(cnpj_limpo) != 14 or not cnpj_limpo.isdigit():
        return "CNPJ Inválido"
    return f"{cnpj_limpo[:2]}.***.***/{cnpj_limpo[8:12]}-{cnpj_limpo[12:]}"

def mascarar_documento_fiscal(doc_str: Optional[str]) -> str:
    """Detecta CPF ou CNPJ e aplica máscara automaticamente."""
    if not doc_str:
        return ""
    doc_limpo = re.sub(r"\D", "", doc_str)
    if len(doc_limpo) == 11:
        return mascarar_cpf(doc_limpo)
    if len(doc_limpo) == 14:
        return mascarar_cnpj(doc_limpo)
    return f"{doc_str[:4]}..." if len(doc_str) > 4 else doc_str

# --- Funções de Criptografia ---
def gerar_chave() -> Optional[bytes]:
    """Gera uma chave simétrica (Fernet). Retorna None se cryptography não estiver disponível."""
    if not CRYPTO_OK:
        log.warning("Tentativa de gerar chave sem biblioteca 'cryptography'.")
        return None
    key = Fernet.generate_key()
    log.info("Nova chave de criptografia gerada com sucesso.")
    return key

def carregar_chave_do_env(var_name: str = "APP_SECRET_KEY") -> Optional[bytes]:
    """
    Carrega ou gera automaticamente a chave de criptografia.
    - Lê do ambiente (ex: export APP_SECRET_KEY='<base64>').
    - Se não existir, gera nova e define em os.environ (ideal para Streamlit Cloud).
    """
    if not CRYPTO_OK:
        log.warning("Criptografia indisponível (módulo 'cryptography' ausente).")
        return None

    val = os.getenv(var_name)
    if val:
        val = val.strip()
        try:
            Fernet(val.encode("utf-8"))
            log.info(f"Chave de criptografia carregada do ambiente ({var_name}).")
            return val.encode("utf-8")
        except Exception:
            log.warning(f"Chave inválida em {var_name}. Tentando decodificar base64...")

            try:
                raw = base64.urlsafe_b64encode(base64.urlsafe_b64decode(val))
                Fernet(raw)
                log.info("Chave base64 decodificada e validada com sucesso.")
                return raw
            except Exception:
                log.error("Falha ao validar chave existente. Gerando nova.")
                key = gerar_chave()
                if key:
                    os.environ[var_name] = key.decode()
                return key

    # Se não houver chave, gera uma nova automaticamente
    log.warning(f"Nenhuma chave encontrada no ambiente ({var_name}). Gerando nova.")
    key = gerar_chave()
    if key:
        os.environ[var_name] = key.decode()
        log.info(f"Nova chave armazenada temporariamente no ambiente ({var_name}).")
    return key


class Cofre:
    """
    Wrapper para criptografia/descriptografia com Fernet.
    Opera de forma segura, mesmo se a biblioteca 'cryptography' estiver ausente.
    """
    def __init__(self, key: Optional[bytes] = None) -> None:
        if not CRYPTO_OK:
            self.available = False
            self._fernet = None
            log.warning("Cofre inicializado SEM suporte à criptografia (cryptography ausente).")
            return

        key = key or carregar_chave_do_env()
        if not key:
            self.available = False
            self._fernet = None
            log.warning("Cofre iniciado sem chave — criptografia desativada.")
            return

        try:
            self._fernet = Fernet(key)
            self.available = True
            log.info("Cofre inicializado com chave válida.")
        except Exception as e:
            self.available = False
            self._fernet = None
            log.error(f"Erro ao inicializar Cofre: {e}")

    # --- Métodos de criptografia ---
    def encrypt_bytes(self, data: bytes) -> bytes:
        if not self.available or not self._fernet:
            return data
        try:
            return self._fernet.encrypt(data)
        except Exception as e:
            log.error(f"Falha ao criptografar bytes: {e}")
            return data

    def decrypt_bytes(self, token: bytes) -> bytes:
        if not self.available or not self._fernet:
            return token
        try:
            return self._fernet.decrypt(token)
        except Exception as e:
            log.warning(f"Falha ao descriptografar token: {e}. Retornando dados originais.")
            return token

    def encrypt_text(self, text: str) -> str:
        """Criptografa texto e retorna em base64."""
        if not text:
            return text
        if not self.available or not self._fernet:
            return text
        try:
            out = self._fernet.encrypt(text.encode("utf-8"))
            return base64.b64encode(out).decode("utf-8")
        except Exception as e:
            log.error(f"Erro ao criptografar texto: {e}")
            return text

    def decrypt_text(self, text: str) -> str:
        """Descriptografa texto em base64 e retorna original."""
        if not text:
            return text
        if not self.available or not self._fernet:
            return text
        try:
            raw = base64.b64decode(text.encode("utf-8"))
            plain = self._fernet.decrypt(raw)
            return plain.decode("utf-8")
        except (base64.binascii.Error, TypeError):
            return text  # já é texto puro
        except Exception as e:
            log.warning(f"Falha ao descriptografar texto '{text[:10]}...': {e}")
            return text
