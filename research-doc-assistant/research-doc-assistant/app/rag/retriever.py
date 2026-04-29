"""混合检索模块 - 稠密向量 + BM25 稀疏检索多路召回

技术要点：
- 稠密向量检索（语义匹配）
- BM25 稀疏检索（关键词精确匹配，适配科研术语）
- RRF（Reciprocal Rank Fusion）多路结果融合
- 中文分词适配学术文本
"""

from __future__ import annotations

from langchain_core.documents import Document
from loguru import logger


class BM25Retriever:
    """BM25 稀疏检索器 - 基于关键词的精确检索"""

    def __init__(self, documents: list[Document] | None = None):
        self.documents: list[Document] = documents or []
        self._index = None

    def add_documents(self, documents: list[Document]) -> None:
        self.documents.extend(documents)
        self._index = None  # 重建索引

    def _build_index(self):
        if self._index is not None:
            return

        from rank_bm25 import BM25Okapi

        from app.document.parser import AcademicTextCleaner

        cleaner = AcademicTextCleaner()
        # 中文分词处理
        tokenized_corpus = [
            cleaner.segment_chinese(doc.page_content).split()
            for doc in self.documents
        ]
        self._index = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int = 10) -> list[Document]:
        if not self.documents:
            return []

        self._build_index()

        from app.document.parser import AcademicTextCleaner

        cleaner = AcademicTextCleaner()
        tokenized_query = cleaner.segment_chinese(query).split()

        scores = self._index.get_scores(tokenized_query)

        # 取 top-k
        scored_docs = list(zip(scores, self.documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scored_docs[:k]:
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "bm25_score": float(score)},
            )
            results.append(doc_copy)

        return results


class HybridRetriever:
    """混合检索器 - 稠密向量检索 + BM25 稀疏检索 + RRF 融合

    融合策略：RRF (Reciprocal Rank Fusion)
    score(d) = Σ 1 / (k + rank_i(d))  其中 k=60 (常数)
    """

    def __init__(
        self,
        vector_store_manager=None,
        bm25_retriever: BM25Retriever | None = None,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        self.vector_store_manager = vector_store_manager
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    async def retrieve(self, query: str, k: int = 10) -> list[Document]:
        """执行混合检索并融合结果"""
        # 1. 稠密向量检索
        dense_results = await self.vector_store_manager.similarity_search(
            query, k=k * 2
        )
        logger.debug(f"稠密检索召回 {len(dense_results)} 个结果")

        # 2. BM25 稀疏检索
        sparse_results = self.bm25_retriever.search(query, k=k * 2)
        logger.debug(f"BM25 检索召回 {len(sparse_results)} 个结果")

        # 3. RRF 融合
        fused = self._rrf_fusion(dense_results, sparse_results, k=k)
        logger.info(f"混合检索完成: 融合后 {len(fused)} 个结果")
        return fused

    def _rrf_fusion(
        self,
        dense_results: list[Document],
        sparse_results: list[Document],
        k: int,
    ) -> list[Document]:
        """RRF 排序融合"""
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        # 稠密检索结果打分
        for rank, doc in enumerate(dense_results):
            doc_key = doc.page_content[:200]  # 用内容前缀作为去重键
            score = self.dense_weight / (self.rrf_k + rank + 1)
            doc_scores[doc_key] = doc_scores.get(doc_key, 0) + score
            doc_map[doc_key] = doc

        # 稀疏检索结果打分
        for rank, doc in enumerate(sparse_results):
            doc_key = doc.page_content[:200]
            score = self.sparse_weight / (self.rrf_k + rank + 1)
            doc_scores[doc_key] = doc_scores.get(doc_key, 0) + score
            if doc_key not in doc_map:
                doc_map[doc_key] = doc

        # 按融合分数排序
        sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

        results = []
        for key in sorted_keys[:k]:
            doc = doc_map[key]
            doc.metadata["rrf_score"] = doc_scores[key]
            results.append(doc)

        return results
