# agentes/llm_mapper.py

from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, Optional
from .utils import textual_truncate  

try:
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.language_models.chat_models import BaseChatModel
except Exception as _e:
    raise ImportError("AgenteLLMMapper requer langchain-core. Instale com `pip install langchain-core`.") from _e

log = logging.getLogger("projeto_fiscal.agentes")

class AgenteLLMMapper:
    """
    Mapeia campos fiscais a partir de texto OCR usando LLM (opcional).
    Retorna dicionário com todos os campos inferidos + scores em `__meta__`.
    """

    def __init__(self, llm: Optional[BaseChatModel]):
        self.llm = llm

    def mapear(self, texto: str, nome_arquivo: str = "") -> Dict[str, Any]:
        if not self.llm or not texto or len(texto) < 20:
            return {}

        schema_campos = [
            # Identificação e chave
            "chave_acesso","numero_nota","serie","modelo","data_emissao","data_saida","hora_emissao","hora_saida",
            # Emitente
            "emitente_nome","emitente_cnpj","emitente_cpf","emitente_ie","emitente_im",
            "emitente_endereco","emitente_municipio","emitente_uf",
            # Destinatário / Tomador
            "destinatario_nome","destinatario_cnpj","destinatario_cpf","destinatario_ie","destinatario_im",
            "destinatario_endereco","destinatario_municipio","destinatario_uf",
            # Totais
            "valor_total","valor_produtos","valor_servicos","valor_icms","valor_ipi","valor_pis","valor_cofins","valor_iss",
            "desconto_total","outras_despesas","frete",
            # Gerais
            "uf","municipio","inscricao_estadual","endereco","cfop","ncm","cst","natureza_operacao",
            # Extras
            "forma_pagamento","cnpj_autorizado","observacoes",
        ]

        sys = SystemMessage(content=(
            "Você é um especialista em leitura fiscal brasileira. "
            "Extraia **campos estruturados** de uma nota (NFe, NFCe, NFSe ou CTe) a partir do texto OCR.\n\n"
            "⚙️ Regras:\n"
            "- Responda **apenas** com um JSON válido (sem explicações).\n"
            "- Use exatamente as chaves do schema fornecido.\n"
            "- Ausente => null.\n"
            "- Datas em YYYY-MM-DD; valores em número (sem 'R$').\n"
            "- Inclua `__meta__` com confidences (0..1) por campo.\n"
            "- Não invente dados."
        ))

        user = HumanMessage(content=(
            f"Arquivo: {nome_arquivo}\n"
            f"Texto OCR:\n{textual_truncate(texto, 4000)}\n\n"
            f"Campos esperados (JSON): {json.dumps(schema_campos, ensure_ascii=False)}\n\n"
            "Responda APENAS com o JSON contendo os campos e o objeto __meta__."
        ))

        try:
            resposta = self.llm.invoke([sys, user]).content.strip()
            m = re.search(r"\{.*\}", resposta, re.S)
            if not m:
                log.warning("LLMMapper: resposta sem JSON detectável.")
                return {}

            payload = json.loads(m.group(0))
            if not isinstance(payload, dict):
                log.warning("LLMMapper: JSON raiz não é objeto.")
                return {}

            resultado = {k: payload.get(k, None) for k in schema_campos}
            meta = payload.get("__meta__", {})
            resultado["__meta__"] = meta if isinstance(meta, dict) else {}
            return resultado

        except Exception as e:
            log.warning(f"LLMMapper falhou: {e}")
            return {}


__all__ = ["AgenteLLMMapper"]
