"""文本分块模块 - 递归分块 + 学术文档语义分块

技术要点：
- 递归字符分块（RecursiveCharacterTextSplitter）
- 学术文档专用语义分块（按章节、段落、公式边界切分）
- 元数据保留与传播
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


class AcademicChunker:
    """学术文档专用语义分块器"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 学术文档分隔符优先级：章节 > 段落 > 句子 > 短语
        self.separators = separators or [
            "\n## ",       # 二级标题
            "\n### ",      # 三级标题
            "\n\n",        # 段落
            "。",           # 中文句号
            ".\n",         # 英文句尾+换行
            ". ",          # 英文句号
            "；",           # 中文分号
            "; ",          # 英文分号
            "\n",          # 换行
            " ",           # 空格
        ]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """对文档列表进行语义分块"""
        all_chunks: list[Document] = []

        for doc in documents:
            content_type = doc.metadata.get("content_type", "text")

            # 表格和公式不分块
            if content_type in ("table", "formula"):
                all_chunks.append(doc)
                continue

            chunks = self._splitter.split_documents([doc])

            # 添加分块索引到元数据
            for idx, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = idx
                chunk.metadata["total_chunks"] = len(chunks)

            all_chunks.extend(chunks)

        logger.info(
            f"语义分块完成: {len(documents)} 个文档 → {len(all_chunks)} 个块 "
            f"(chunk_size={self.chunk_size})"
        )
        return all_chunks


class HyDEGenerator:
    """HyDE（假设文档嵌入）- 生成假设性答案文档以改善检索质量

    技术原理：
    对于用户查询，先用 LLM 生成一段"假设性回答"，再将该回答向量化
    进行检索，利用假设回答与真实文档的语义接近性提升召回率。
    """

    PROMPT_TEMPLATE = """请根据以下学术问题，撰写一段简短的假设性回答（约100-200字），
该回答应当像是从一篇科研论文中摘录的片段。请直接输出回答内容，不要包含任何前缀。

问题：{question}

假设性回答："""

    def __init__(self, llm=None):
        from app.core.llm_factory import get_llm

        self.llm = llm or get_llm()

    async def generate(self, question: str) -> str:
        """生成假设文档"""
        from langchain_core.messages import HumanMessage

        prompt = self.PROMPT_TEMPLATE.format(question=question)
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content
