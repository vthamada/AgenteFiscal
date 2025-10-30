# agentes/utils.py

from __future__ import annotations
from typing import Optional
import re
import logging

# ---------------- Logger único dos agentes ----------------
log = logging.getLogger("agente_fiscal.agentes")
if not log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)

# --------------- Constantes/regex e helpers ---------------
_WHITESPACE_RE = re.compile(r"\s+", re.S)
_MONEY_CHARS_RE = re.compile(r"[^\d\.,\-]")
_UF_SET = {
    "AC","AL","AP","AM","BA","CE","DF","ES","GO",
    "MA","MT","MS","MG","PA","PB","PR","PE","PI",
    "RJ","RN","RS","RO","RR","SC","SP","SE","TO",
}

def _norm_ws(texto: str) -> str:
    return _WHITESPACE_RE.sub(" ", (texto or "").strip())

def _only_digits(s: Optional[str]) -> Optional[str]:
    return re.sub(r"\D+", "", s) if s else None

def _to_float_br(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s2 = str(s).strip()
    # Remove tudo exceto dígitos, pontos, vírgulas e sinal de negativo
    s2 = _MONEY_CHARS_RE.sub("", s2)
    # Normalização heurística:
    # - Se houver tanto '.' quanto ',' e o '.' aparece antes da ',' -> interpretamos '.' como separador de milhares e ',' como decimal (formato BR)
    # - Se houver tanto '.' quanto ',' e a ',' aparece antes do '.' -> interpretamos ',' como separador de milhares e '.' como decimal (menos comum)
    # - Se houver apenas ',' -> é decimal (BR) -> substituir por '.'
    # - Se houver apenas '.' -> é decimal (EN) -> manter
    try:
        if "," in s2 and "." in s2:
            if s2.rfind(".") < s2.rfind(","):
                # ex: 1.234,56 -> 1234.56
                s2 = s2.replace(".", "").replace(",", ".")
            else:
                # ex: 1,234.56 -> 1234.56
                s2 = s2.replace(",", "")
        elif "," in s2:
            # ex: 1234,56 -> 1234.56
            s2 = s2.replace(".", "").replace(",", ".")
        else:
            # somente ponto(s) ou apenas dígitos: remove possíveis espaços já removidos
            s2 = s2.replace(",", "")
        return float(s2)
    except Exception:
        return None

def _parse_date_like(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    m = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})(?:[ T]\d{2}:\d{2}:\d{2})?", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"(\d{2})[-/](\d{2})[-/](\d{4})(?:[ T]\d{2}:\d{2}:\d{2})?", s)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return None

def _safe_title(x: Optional[str]) -> Optional[str]:
    if not x:
        return x
    xt = _norm_ws(x)
    try:
        if len(xt) <= 4 and xt.upper() in _UF_SET:
            return xt.upper()
        return xt
    except Exception:
        return xt

def _clamp(v: Optional[float], lo: float, hi: float) -> Optional[float]:
    if v is None:
        return None
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return None

def textual_truncate(s: str, max_len: int) -> str:
    s = s or ""
    if len(s) <= max_len:
        return s
    head = s[: max_len // 2]
    tail = s[-max_len // 2 :]
    return head + "\n...\n" + tail

__all__ = [
    "log",
    "_norm_ws","_only_digits","_to_float_br","_parse_date_like",
    "_safe_title","_clamp","textual_truncate","_UF_SET",
]
