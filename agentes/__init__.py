# __init__.py

from .agente_xml_parser import AgenteXMLParser
from .agente_normalizador import AgenteNormalizadorCampos
from .agente_associador_xml import AgenteAssociadorXML
from .metrics_agent import MetricsAgent

# Agente analítico (opcional; depende de langchain-core)
try:
    from .agente_analitico import AgenteAnalitico  # type: ignore
    _ANALYTICS_AVAILABLE = True
except Exception:
    AgenteAnalitico = None  # type: ignore
    _ANALYTICS_AVAILABLE = False

# Utils e constantes comuns
from .utils import log, _UF_SET
from .utils import _to_float_br
import re

# Removido: utilitários específicos de OCR/NLP não são mais expostos neste pacote.

# Core availability flag (mantém compatibilidade com código antigo)
CORE_MODULES_AVAILABLE = True

__all__ = [
    "AgenteXMLParser",
    "AgenteNormalizadorCampos",
    "AgenteAssociadorXML",
    "MetricsAgent",
    "CORE_MODULES_AVAILABLE",
    "log",
    "_UF_SET",
]

# Exporta AgenteAnalitico apenas quando disponível
if _ANALYTICS_AVAILABLE:
    __all__.append("AgenteAnalitico")
