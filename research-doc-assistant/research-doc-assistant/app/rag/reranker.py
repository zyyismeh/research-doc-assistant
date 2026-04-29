"""重排序与上下文压缩模块

技术要点：
- BGE-reranker 交叉编码器重排模型
- 上下文压缩（去除与查询无关的信息）
- 检索结果校验（过滤低质量文档）
"""

from __future__ import annotations

from langchain_core.documents import Document
from loguru import logger

from app.core import settings


class BGEReranker:
    """BGE-reranker 模型重排序

    利用交叉编码器对 query-document 对进行精细打分，
    相比向量检索的双编码器有更高的精度。
    """

    def __init__(self, model_name: str | None = None, top_k: int = 5):
        self.model_name = model_name or settings.reranker_model_name
        self.top_k = top_k
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker

                self._model = FlagReranker(
                    self.model_name, use_fp16=True
                )
                logger.info(f"BGE Reranker 模型加载完成: {self.model_name}")
            except ImportError:
                logger.warning(
                    "FlagEmbedding 未安装，使用 LLM 备选重排方案"
                )
                self._model = "fallback"
        return self._model

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """对检索结果重排序"""
        if not documents:
            return []

        if self.model == "fallback":
            return self._fallback_rerank(query, documents)

        # 构建 query-document 对
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.compute_score(pairs, normalize=True)

        # 如果只有一个文档，scores 不是列表
        if isinstance(scores, (int, float)):
            scores = [scores]

        # 按分数排序
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scored_docs[: self.top_k]:
            doc.metadata["rerank_score"] = float(score)
            results.append(doc)

        logger.info(
            f"重排序完成: {len(documents)} → {len(results)} 个文档, "
            f"最高分: {results[0].metadata.get('rerank_score', 0):.4f}"
        )
        return results

    def _fallback_rerank(
        self, query: str, documents: list[Document]
    ) -> list[Document]:
        """备选方案: 基于关键词重叠度简单重排"""
        query_terms = set(query.lower().split())

        scored = []
        for doc in documents:
            doc_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms & doc_terms)
            scored.append((overlap, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[: self.top_k]]


class ContextCompressor:
    """上下文压缩 - 提取文档中与查询最相关的段落"""

    COMPRESS_PROMPT = """从以下文档片段中，提取与问题最相关的核心内容。
只保留直接回答问题或提供关键证据的部分，去除无关信息。
输出压缩后的文本，不要添加任何说明。

问题：{question}

文档片段：
{context}

压缩后的核心内容："""

    def __init__(self, llm=None):
        from app.core.llm_factory import get_llm

        self.llm = llm or get_llm()

    async def compress(
        self, query: str, documents: list[Document]
    ) -> list[Document]:
        """压缩文档上下文"""
        from langchain_core.messages import HumanMessage

        compressed = []
        for doc in documents:
            # 短文本不需要压缩
            if len(doc.page_content) < 300:
                compressed.append(doc)
                continue

            prompt = self.COMPRESS_PROMPT.format(
                question=query, context=doc.page_content
            )
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])

            compressed_doc = Document(
                page_content=response.content,
                metadata={**doc.metadata, "compressed": True},
            )
            compressed.append(compressed_doc)

        logger.info(f"上下文压缩完成: {len(compressed)} 个文档")
        return compressed


class ResultValidator:
    """检索结果校验 - 过滤低质量、不相关的检索结果"""

    def __init__(self, min_score: float = 0.1, min_length: int = 20):
        self.min_score = min_score
        self.min_length = min_length

    def validate(self, documents: list[Document]) -> list[Document]:
        """校验并过滤检索结果"""
        valid = []
        for doc in documents:
            # 过滤过短的文档
            if len(doc.page_content.strip()) < self.min_length:
                continue
            # 过滤重排分数过低的文档
            rerank_score = doc.metadata.get("rerank_score")
            if rerank_score is not None and rerank_score < self.min_score:
                continue
            valid.append(doc)

        filtered_count = len(documents) - len(valid)
        if filtered_count > 0:
            logger.info(f"结果校验: 过滤 {filtered_count} 个低质量文档")

        return valid
