"""RAG 核心链路 - Prompt 工程 + 链路编排

技术要点：
- 学术问答专用 Prompt 模板
- 思维链（CoT）推理
- 摘要生成链路
- 多文档跨文献检索
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ===================== Prompt 模板 =====================

ACADEMIC_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一位专业的科研文档助手，擅长分析学术文献并回答科研问题。
请基于提供的参考文献内容回答问题。

要求：
1. 回答必须基于给定的参考文献，不要编造内容
2. 引用关键信息时标注来源文档
3. 如果参考文献中没有直接答案，请说明并给出基于文献的推理
4. 使用学术化的语言风格
5. 对于涉及数据的问题，请精确引用原文数据""",
        ),
        (
            "human",
            """参考文献内容：
{context}

问题：{question}

请给出详细、准确的回答：""",
        ),
    ]
)

ACADEMIC_QA_COT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一位专业的科研文档助手。请使用思维链（Chain of Thought）方法逐步分析问题。

分析步骤：
1. 【理解问题】明确问题的核心要求
2. 【文献检索】从参考文献中定位相关信息
3. 【逻辑推理】基于文献内容进行逻辑推理
4. 【综合回答】给出完整、准确的答案
5. 【来源标注】标注关键信息的出处""",
        ),
        (
            "human",
            """参考文献内容：
{context}

问题：{question}

请按步骤分析并回答：""",
        ),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一位学术摘要生成专家。请对以下科研文档内容生成结构化摘要。

摘要结构：
1. 【研究背景】研究的背景及意义
2. 【研究方法】使用的方法和技术路线
3. 【主要发现】核心实验结果和发现
4. 【结论】主要结论和贡献
5. 【关键词】3-5个关键词""",
        ),
        (
            "human",
            """文档内容：
{context}

请生成学术摘要：""",
        ),
    ]
)

VALIDITY_CHECK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一位学术质量审核专家。请评估以下回答的质量。

评估维度：
1. 事实准确性：回答是否与参考文献一致
2. 完整性：是否覆盖了问题的所有方面
3. 逻辑性：推理过程是否合理
4. 学术规范：语言和引用是否规范

请给出 1-10 的质量评分和简要评价。如果评分低于 6 分，请指出需要改进的具体问题。""",
        ),
        (
            "human",
            """原始问题：{question}

参考文献：{context}

生成的回答：{answer}

请评估：""",
        ),
    ]
)


def format_docs(docs: list[Document]) -> str:
    """将文档列表格式化为上下文字符串"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", "未知来源")
        page = doc.metadata.get("page", "")
        section = doc.metadata.get("section_title", "")

        header = f"[文献{i}] {source}"
        if page:
            header += f" (第{page}页)"
        if section:
            header += f" - {section}"

        formatted.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


class RAGChain:
    """RAG 核心问答链"""

    def __init__(self, llm=None):
        from app.core.llm_factory import get_llm

        self.llm = llm or get_llm()
        self.output_parser = StrOutputParser()

    def build_qa_chain(self, use_cot: bool = False):
        """构建学术问答链"""
        prompt = ACADEMIC_QA_COT_PROMPT if use_cot else ACADEMIC_QA_PROMPT
        return prompt | self.llm | self.output_parser

    def build_summary_chain(self):
        """构建摘要生成链"""
        return SUMMARY_PROMPT | self.llm | self.output_parser

    def build_validity_chain(self):
        """构建质量校验链"""
        return VALIDITY_CHECK_PROMPT | self.llm | self.output_parser

    async def answer(
        self,
        question: str,
        context_docs: list[Document],
        use_cot: bool = True,
    ) -> str:
        """执行学术问答"""
        context = format_docs(context_docs)
        chain = self.build_qa_chain(use_cot=use_cot)
        return await chain.ainvoke({"question": question, "context": context})

    async def summarize(self, documents: list[Document]) -> str:
        """生成文档摘要"""
        context = format_docs(documents)
        chain = self.build_summary_chain()
        return await chain.ainvoke({"context": context})

    async def validate_answer(
        self,
        question: str,
        context_docs: list[Document],
        answer: str,
    ) -> str:
        """校验回答质量"""
        context = format_docs(context_docs)
        chain = self.build_validity_chain()
        return await chain.ainvoke(
            {"question": question, "context": context, "answer": answer}
        )
