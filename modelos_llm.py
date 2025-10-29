# modelos_llm.py
from __future__ import annotations

import os
import time
import json
from typing import Optional, Literal, List, Dict, Any, Tuple

# Seus agentes usam BaseChatModel de langchain-core
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

# Provedores
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================
# Catálogos de modelos (somente referência)
# ==============================
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.5-pro",
]

OPENAI_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4o",
    "gpt-4.1",
]

OPENROUTER_MODELS = [
    "anthropic/claude-3.7-sonnet",
    "google/gemini-2.5-pro",
    "openai/gpt-5",
    "qwen/qwen3-vl-235b-a22b-thinking",
    "deepseek/deepseek-v3.1",
    "xai/grok-4-fast",
]

# ==============================
# Prompts canônicos
# ==============================
PROMPT_OCR_CORRECTOR = (
    "Você corrige ruído de OCR em documentos fiscais brasileiros sem inventar dados. "
    "Reescreva apenas para normalizar espaçamentos, pontuação e erros óbvios (ex: VAL0R→VALOR, EMISSA0→EMISSAO). "
    "Mantenha números e siglas como estão. Responda apenas o texto corrigido."
)

PROMPT_NLP_EXTRACTOR = (
    "Você é especialista em NFe/NFCe/NFSe/CTe do Brasil. Extraia um JSON **estrito** com as chaves fornecidas. "
    "Regras:\n"
    "- Datas em YYYY-MM-DD.\n"
    "- Valores numéricos com ponto decimal (ex.: 1234.56).\n"
    "- Se um campo não existir, use null.\n"
    "- Não invente dados. Responda **apenas** JSON válido."
)

PROMPT_NORMALIZER_REFINER = (
    "Você normaliza nomes/endereços mantendo sentido fiscal. Ajuste apenas caixa/acentuação e espaços duplicados. "
    "Não crie nem remova termos. Retorne JSON com as MESMAS chaves recebidas."
)

PROMPT_ASSOC_XML = (
    "Você recebe o texto de uma nota (OCR) e o XML correspondente. "
    "Associe campos por similaridade semântica (OCR↔XML) e produza um JSON com mapeamentos e um score 0..1 por campo. "
    "Não invente valores; se não houver correspondência, use null."
)

PROMPT_VALIDATION_EXPLAINER = (
    "Explique de forma breve e técnica inconsistências fiscais encontradas, em português, "
    "sugerindo correções objetivas quando aplicável. Retorne uma lista JSON de mensagens."
)

# ==============================
# Utilitários internos
# ==============================
def _first_json_in_text(text: str) -> Any:
    """Retorna o primeiro bloco { ... } parseável como JSON; {} em caso de falha."""
    try:
        import re
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}

def _token_estimate(*parts: str) -> Tuple[int, int]:
    """Estimativa grosseira de tokens (chars/4) p/ input/output."""
    total_chars = sum(len(p or "") for p in parts)
    return max(1, total_chars // 4), 0

def _confidence_heuristic_ok_keys(payload: Dict[str, Any], expected_keys: Optional[List[str]]) -> float:
    """Heurística simples: proporção de chaves esperadas presentes e não-nulas."""
    if not expected_keys:
        return 0.7 if payload else 0.3
    if not payload:
        return 0.0
    ok = 0
    for k in expected_keys:
        v = payload.get(k)
        if v not in (None, "", [], {}):
            ok += 1
    return round(ok / max(1, len(expected_keys)), 3)

def _get_model_name(llm: BaseChatModel) -> str:
    for attr in ("model", "model_name", "_model", "model_id"):
        v = getattr(llm, attr, None)
        if v:
            return str(v)
    return "unknown"

def _get_provider(llm: BaseChatModel) -> str:
    base_url = getattr(llm, "base_url", None)
    if base_url and "openrouter" in str(base_url).lower():
        return "openrouter"
    cls = llm.__class__.__name__.lower()
    if "google" in cls or "gemini" in cls:
        return "gemini"
    if "openai" in cls:
        return "openai"
    return "desconhecido"

def _bind_temperature(llm: BaseChatModel, temperature: Optional[float]) -> BaseChatModel:
    """Tenta llm.bind(temperature=...), senão retorna o próprio llm."""
    if temperature is None:
        return llm
    try:
        return llm.bind(temperature=temperature)  # type: ignore[attr-defined]
    except Exception:
        return llm

# ==============================
# Fábrica de LLMs
# ==============================
def make_llm(
    provider: Literal["gemini", "openai", "openrouter"],
    model: str,
    *,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """
    Cria e configura uma instância BaseChatModel compatível com .invoke([...]).
    """
    provider = (provider or "").lower().strip()

    if provider == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY não informada.")
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=temperature)

    elif provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY não informada.")
        llm = ChatOpenAI(model=model, api_key=key, temperature=temperature)  # type: ignore[arg-type]

    elif provider == "openrouter":
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY não informada.")
        llm = ChatOpenAI(  # type: ignore[call-arg]
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            model=model,
            temperature=temperature,
            max_tokens=4096,
        )
        try:
            setattr(llm, "base_url", "https://openrouter.ai/api/v1")
        except Exception:
            pass
    else:
        raise RuntimeError(f"Provider LLM inválido: {provider!r}")

    # Marca meta direto no objeto
    try:
        setattr(llm, "_provider", provider)
        setattr(llm, "_model_name", model)
        setattr(llm, "_default_temperature", temperature)
    except Exception:
        pass

    return llm

# ==============================
# Wrappers cognitivos (centro)
# ==============================
def invoke_with_context(
    llm: BaseChatModel,
    *,
    system_prompt: str,
    user_prompt: str,
    task_tag: str = "generic",
    temperature: Optional[float] = None,
    json_expected: bool = False,
    expected_keys: Optional[List[str]] = None,
    max_chars_user: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Invoca o LLM (System+User) e retorna:
      { "content": <str>, "json": <dict|None>, "meta": {...}, "raw": <obj> }
    """
    start_s = time.perf_counter()

    if max_chars_user and len(user_prompt or "") > max_chars_user:
        user_prompt = (user_prompt or "")[:max_chars_user]

    llm_call = _bind_temperature(llm, temperature)

    sys_msg = SystemMessage(content=system_prompt)
    usr_msg = HumanMessage(content=user_prompt)

    approx_in, _ = _token_estimate(system_prompt, user_prompt)
    try:
        resp = llm_call.invoke([sys_msg, usr_msg])  # type: ignore[attr-defined]
        content = getattr(resp, "content", None) or str(resp)
        approx_out = max(1, len(content) // 4)
        elapsed = time.perf_counter() - start_s

        payload_json = None
        confidence = None
        if json_expected:
            payload_json = _first_json_in_text(content)
            confidence = _confidence_heuristic_ok_keys(payload_json, expected_keys)

        meta = {
            "provider": getattr(llm, "_provider", _get_provider(llm)),
            "model": getattr(llm, "_model_name", _get_model_name(llm)),
            "used_temperature": temperature if temperature is not None else getattr(llm, "_default_temperature", None),
            "latency_s": round(elapsed, 3),
            "tokens_input~approx": approx_in,
            "tokens_output~approx": approx_out,
            "task": task_tag,
            "confidence": confidence,
        }
        return {"content": content, "json": payload_json, "meta": meta, "raw": resp}

    except Exception as e:
        elapsed = time.perf_counter() - start_s
        meta = {
            "provider": getattr(llm, "_provider", _get_provider(llm)),
            "model": getattr(llm, "_model_name", _get_model_name(llm)),
            "used_temperature": temperature if temperature is not None else getattr(llm, "_default_temperature", None),
            "latency_s": round(elapsed, 3),
            "tokens_input~approx": approx_in,
            "tokens_output~approx": 0,
            "task": task_tag,
            "error": str(e),
        }
        return {"content": "", "json": None, "meta": meta, "raw": None}

def invoke_json_extractor(
    llm: BaseChatModel,
    *,
    schema_keys: List[str],
    text: str,
    extra_instructions: Optional[str] = None,
    temperature: Optional[float] = 0.0,
    max_chars_user: int = 6000,
) -> Dict[str, Any]:
    """
    Extrator JSON canônico para OCR/NLP/Normalização/Associação.
    Retorna: { "json": <dict|{}>, "meta": {...}, "content": <str>, "raw": <obj> }
    """
    instructions = PROMPT_NLP_EXTRACTOR
    if extra_instructions:
        instructions += "\n\nInstruções adicionais: " + extra_instructions

    user_prompt = (
        f"Texto (pode estar truncado):\n{text}\n\n"
        f"Chaves (ordem sugerida): {json.dumps(schema_keys, ensure_ascii=False)}\n"
        f"Responda apenas JSON estrito."
    )

    out = invoke_with_context(
        llm,
        system_prompt=instructions,
        user_prompt=user_prompt,
        task_tag="json_extractor",
        temperature=temperature,
        json_expected=True,
        expected_keys=schema_keys,
        max_chars_user=max_chars_user,
    )
    payload = out.get("json") or {}
    if isinstance(payload, dict):
        for k in schema_keys:
            payload.setdefault(k, None)
    return {**out, "json": payload}

def ocr_correct_text(
    llm: BaseChatModel,
    *,
    noisy_text: str,
    temperature: Optional[float] = 0.0,
    max_chars_user: int = 6000,
) -> Dict[str, Any]:
    """Refinador semântico leve de OCR. Retorna {content, meta, raw}."""
    return invoke_with_context(
        llm,
        system_prompt=PROMPT_OCR_CORRECTOR,
        user_prompt=noisy_text,
        task_tag="ocr_corrector",
        temperature=temperature,
        json_expected=False,
        max_chars_user=max_chars_user,
    )

def normalize_text_fields(
    llm: BaseChatModel,
    *,
    fields_payload: Dict[str, Any],
    temperature: Optional[float] = 0.0,
) -> Dict[str, Any]:
    """Normalizador leve (nomes/endereços). Espera JSON de entrada e saída."""
    user_prompt = json.dumps(fields_payload, ensure_ascii=False)
    return invoke_with_context(
        llm,
        system_prompt=PROMPT_NORMALIZER_REFINER,
        user_prompt=user_prompt,
        task_tag="normalizer_refine",
        temperature=temperature,
        json_expected=True,
        expected_keys=list(fields_payload.keys()),
        max_chars_user=4000,
    )

def associate_ocr_xml(
    llm: BaseChatModel,
    *,
    ocr_text: str,
    xml_text: str,
    temperature: Optional[float] = 0.0,
    max_chars_user: int = 8000,
) -> Dict[str, Any]:
    """Associação semântica OCR↔XML (retorna JSON com mapeamentos e 'score' por campo)."""
    user_prompt = (
        "=== OCR ===\n" + ocr_text + "\n\n"
        "=== XML ===\n" + xml_text + "\n\n"
        "Responda apenas JSON com chaves encontradas, seus valores e score 0..1."
    )
    return invoke_with_context(
        llm,
        system_prompt=PROMPT_ASSOC_XML,
        user_prompt=user_prompt,
        task_tag="associate_ocr_xml",
        temperature=temperature,
        json_expected=True,
        expected_keys=None,  # estrutura livre
        max_chars_user=max_chars_user,
    )

def validation_explanations(
    llm: BaseChatModel,
    *,
    issues_payload: Dict[str, Any],
    temperature: Optional[float] = 0.0,
) -> Dict[str, Any]:
    """Gera explicações curtas em português para inconsistências fiscais."""
    user_prompt = json.dumps(issues_payload, ensure_ascii=False)
    return invoke_with_context(
        llm,
        system_prompt=PROMPT_VALIDATION_EXPLAINER,
        user_prompt=user_prompt,
        task_tag="validation_explainer",
        temperature=temperature,
        json_expected=True,
        expected_keys=None,
        max_chars_user=4000,
    )

# ==============================
# Compatibilidade retroativa (shims)
# ==============================
def get_llm_identity(llm: BaseChatModel) -> Dict[str, Any]:
    """
    Retorna identidade do modelo no formato esperado por agentes legados.
    """
    return {
        "provider": getattr(llm, "_provider", _get_provider(llm)),
        "model": getattr(llm, "_model_name", _get_model_name(llm)),
        "temperature": getattr(llm, "_default_temperature", None),
    }

def ocr_correct_text_compat(
    llm: BaseChatModel,
    *,
    noisy_text: str,
    temperature: Optional[float] = 0.0,
    max_chars_user: int = 6000,
) -> Dict[str, Any]:
    """
    Compat com agentes que esperam {"text": ..., "confidence": ...}.
    """
    out = ocr_correct_text(
        llm,
        noisy_text=noisy_text,
        temperature=temperature,
        max_chars_user=max_chars_user,
    )
    return {
        "text": out.get("content", ""),
        "confidence": (out.get("meta") or {}).get("confidence", 1.0),
        "meta": out.get("meta"),
        "raw": out.get("raw"),
    }

# Aliases de nome para agentes que chamam wrappers antigos por nome
def extract_header_json(llm: BaseChatModel, *, schema_keys: List[str], text: str,
                        extra_instructions: Optional[str] = None,
                        temperature: Optional[float] = 0.0,
                        max_chars_user: int = 6000) -> Dict[str, Any]:
    return invoke_json_extractor(
        llm,
        schema_keys=schema_keys,
        text=text,
        extra_instructions=extra_instructions,
        temperature=temperature,
        max_chars_user=max_chars_user,
    )

def extract_items_json(llm: BaseChatModel, *, schema_keys: List[str], text: str,
                       extra_instructions: Optional[str] = None,
                       temperature: Optional[float] = 0.0,
                       max_chars_user: int = 6000) -> Dict[str, Any]:
    return invoke_json_extractor(
        llm,
        schema_keys=schema_keys,
        text=text,
        extra_instructions=extra_instructions,
        temperature=temperature,
        max_chars_user=max_chars_user,
    )

def extract_taxes_json(llm: BaseChatModel, *, schema_keys: List[str], text: str,
                       extra_instructions: Optional[str] = None,
                       temperature: Optional[float] = 0.0,
                       max_chars_user: int = 6000) -> Dict[str, Any]:
    return invoke_json_extractor(
        llm,
        schema_keys=schema_keys,
        text=text,
        extra_instructions=extra_instructions,
        temperature=temperature,
        max_chars_user=max_chars_user,
    )

def explain_fields(llm: BaseChatModel, *, issues_payload: Dict[str, Any],
                   temperature: Optional[float] = 0.0) -> Dict[str, Any]:
    return validation_explanations(
        llm,
        issues_payload=issues_payload,
        temperature=temperature,
    )

__all__ = [
    # fábrica
    "make_llm",
    # wrappers centrais
    "invoke_with_context",
    "invoke_json_extractor",
    "ocr_correct_text",
    "normalize_text_fields",
    "associate_ocr_xml",
    "validation_explanations",
    # compat retroativa
    "get_llm_identity",
    "ocr_correct_text_compat",
    "extract_header_json",
    "extract_items_json",
    "extract_taxes_json",
    "explain_fields",
    # catálogos (opcional)
    "GEMINI_MODELS",
    "OPENAI_MODELS",
    "OPENROUTER_MODELS",
]
