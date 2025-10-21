# modelos_llm.py

from __future__ import annotations
from typing import Optional, Literal
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.5-pro"
]

OPENAI_MODELS = [
    "gpt-5",                
    "gpt-5-mini",          
    "gpt-5-nano",         
    "gpt-4o",               
    "gpt-4.1"                   
]

OPENROUTER_MODELS = [
    "anthropic/claude-3.7-sonnet",  
    "google/gemini-2.5-pro",        
    "openai/gpt-5",                 
    "qwen/qwen3-vl-235b-a22b-thinking", 
    "deepseek/deepseek-v3.1",       
    "xai/grok-4-fast",             
]

def make_llm(
    provider: Literal["gemini", "openai", "openrouter"],
    model: str,
    *,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """
    Cria e configura uma instância de um modelo de linguagem (LLM).

    Esta função atua como uma fábrica, selecionando o provedor e o modelo corretos
    com base nos parâmetros fornecidos e configurando-o com a chave de API apropriada.

    Args:
        provider (Literal["gemini", "openai", "openrouter"]): O provedor do serviço de LLM.
        model (str): O identificador do modelo a ser utilizado.
        api_key (Optional[str], optional): A chave de API. Se não for fornecida, busca da variável de ambiente.
        temperature (float, optional): A temperatura do modelo (criatividade). Padrão é 0.0.

    Returns:
        BaseChatModel: Uma instância configurada e pronta para uso do LLM.
    """
    provider = provider.lower()

    if provider == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY não informada.")
        return ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=temperature)

    if provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY não informada.")
        return ChatOpenAI(model=model, api_key=key, temperature=temperature)

    # OpenRouter como fallback
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY não informada.")
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        model=model,
        temperature=temperature,
        max_tokens=4096,
    )
