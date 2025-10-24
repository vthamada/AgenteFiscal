# __init__.py

from .agente_llm_mapper import AgenteLLMMapper
from .agente_xml_parser import AgenteXMLParser
from .agente_ocr import AgenteOCR
from .agente_nlp import AgenteNLP
from .agente_analitico import AgenteAnalitico
from .agente_normalizador import AgenteNormalizadorCampos
from .agente_associador_xml import AgenteAssociadorXML
from .agente_confianca_router import AgenteConfiancaRouter
from .metrics_agent import MetricsAgent

# Utils e constantes comuns
from .utils import log, _UF_SET

# Core availability flag (mantém compatibilidade com código antigo)
CORE_MODULES_AVAILABLE = True

__all__ = [
    "AgenteLLMMapper",
    "AgenteXMLParser",
    "AgenteOCR",
    "AgenteNLP",
    "AgenteAnalitico",
    "AgenteNormalizadorCampos",
    "AgenteAssociadorXML",
    "AgenteConfiancaRouter",
    "MetricsAgent",
    "CORE_MODULES_AVAILABLE",
    "log",
    "_UF_SET",
]
