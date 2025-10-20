# seguranca.py

from __future__ import annotations
from pathlib import Path
from typing import Optional
import hashlib
import os
import base64

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
        self._fernet = Fernet(key) if self.available else None

    def encrypt_bytes(self, data: bytes) -> bytes:
        if not self.available or not self._fernet:
            return data
        return self._fernet.encrypt(data)

    def decrypt_bytes(self, token: bytes) -> bytes:
        if not self.available or not self._fernet:
            return token
        return self._fernet.decrypt(token)

    def encrypt_text(self, text: str) -> str:
        out = self.encrypt_bytes(text.encode("utf-8"))
        return base64.b64encode(out).decode("utf-8") if self.available else text

    def decrypt_text(self, text: str) -> str:
        if not self.available:
            return text
        raw = base64.b64decode(text.encode("utf-8"))
        plain = self.decrypt_bytes(raw)
        return plain.decode("utf-8")
