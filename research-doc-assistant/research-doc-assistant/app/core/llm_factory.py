"""大模型与嵌入模型工厂 - 支持通义千问 / OpenAI / Ollama"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.core import settings


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """根据配置创建 LLM 实例"""
    provider = settings.llm_provider

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model_name,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
            temperature=0.1,
            max_tokens=4096,
        )

    elif provider == "dashscope":
        from langchain_community.chat_models.tongyi import ChatTongyi

        return ChatTongyi(
            model=settings.llm_model_name,
            dashscope_api_key=settings.dashscope_api_key,
            temperature=0.1,
            max_tokens=4096,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model_name,
            base_url=settings.ollama_base_url,
            temperature=0.1,
        )

    raise ValueError(f"不支持的 LLM 提供商: {provider}")


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """创建嵌入模型实例（本地 HuggingFace 模型）"""
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings

    return HuggingFaceBgeEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
