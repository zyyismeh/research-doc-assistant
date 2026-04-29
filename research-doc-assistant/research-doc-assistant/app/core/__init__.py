"""核心配置模块 - 基于 pydantic-settings 管理全局配置"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 大模型 ---
    llm_provider: Literal["openai", "dashscope", "ollama"] = "dashscope"
    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    dashscope_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    llm_model_name: str = "qwen-plus"

    # --- 嵌入 ---
    embedding_model_name: str = "BAAI/bge-large-zh-v1.5"
    reranker_model_name: str = "BAAI/bge-reranker-large"

    # --- 向量库 ---
    vector_store_type: Literal["chroma", "milvus", "faiss"] = "chroma"
    chroma_persist_dir: str = "./data/vectorstore/chroma"
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # --- PostgreSQL ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "research_user"
    postgres_password: str = "research_password"
    postgres_db: str = "research_doc_assistant"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl: int = 3600

    # --- 服务 ---
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = True

    # --- LangSmith ---
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "research-doc-assistant"

    # --- 文件上传 ---
    upload_dir: str = "./data/uploads"
    max_file_size_mb: int = 50

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
