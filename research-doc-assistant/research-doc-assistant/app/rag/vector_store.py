"""向量存储模块 - 支持 Chroma / Milvus / FAISS 多后端

技术要点：
- Chroma 轻量级本地开发
- FAISS 内存级快速检索
- Milvus 分布式高性能生产部署
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from loguru import logger

from app.core import settings


class VectorStoreManager:
    """向量数据库管理器 - 统一封装多后端操作"""

    def __init__(
        self,
        embeddings: Embeddings | None = None,
        store_type: str | None = None,
        collection_name: str = "research_docs",
    ):
        from app.core.llm_factory import get_embeddings

        self.embeddings = embeddings or get_embeddings()
        self.store_type = store_type or settings.vector_store_type
        self.collection_name = collection_name
        self._store: VectorStore | None = None

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = self._create_store()
        return self._store

    def _create_store(self) -> VectorStore:
        if self.store_type == "chroma":
            return self._create_chroma()
        elif self.store_type == "faiss":
            return self._create_faiss()
        elif self.store_type == "milvus":
            return self._create_milvus()
        raise ValueError(f"不支持的向量库类型: {self.store_type}")

    def _create_chroma(self) -> VectorStore:
        from langchain_community.vectorstores import Chroma

        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"初始化 Chroma 向量库: {persist_dir}")
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(persist_dir),
        )

    def _create_faiss(self) -> VectorStore:
        from langchain_community.vectorstores import FAISS

        faiss_path = Path("./data/vectorstore/faiss")
        if (faiss_path / "index.faiss").exists():
            logger.info("加载已有 FAISS 索引")
            return FAISS.load_local(
                str(faiss_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        logger.info("创建新的 FAISS 索引")
        # FAISS 需要初始文档，返回空的占位
        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore
        from langchain_community.vectorstores import FAISS

        embedding_dim = len(self.embeddings.embed_query("test"))
        index = faiss.IndexFlatIP(embedding_dim)
        return FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def _create_milvus(self) -> VectorStore:
        from langchain_community.vectorstores import Milvus

        logger.info(
            f"连接 Milvus: {settings.milvus_host}:{settings.milvus_port}"
        )
        return Milvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args={
                "host": settings.milvus_host,
                "port": settings.milvus_port,
            },
        )

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """添加文档到向量库"""
        logger.info(f"向向量库添加 {len(documents)} 个文档块")
        ids = await self.store.aadd_documents(documents)
        # FAISS 需要手动持久化
        if self.store_type == "faiss":
            faiss_path = Path("./data/vectorstore/faiss")
            faiss_path.mkdir(parents=True, exist_ok=True)
            self.store.save_local(str(faiss_path))
        return ids

    async def similarity_search(
        self, query: str, k: int = 10
    ) -> list[Document]:
        """稠密向量相似度检索"""
        return await self.store.asimilarity_search(query, k=k)

    async def similarity_search_with_score(
        self, query: str, k: int = 10
    ) -> list[tuple[Document, float]]:
        """带分数的相似度检索"""
        return await self.store.asimilarity_search_with_score(query, k=k)

    def as_retriever(self, search_kwargs: dict | None = None):
        """转换为 LangChain Retriever"""
        return self.store.as_retriever(
            search_kwargs=search_kwargs or {"k": 10}
        )
