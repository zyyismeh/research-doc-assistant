"""LangGraph 多步骤科研文档工作流

工作流：检索 → 重排 → 摘要生成 → 学术问答 → 结果校验
技术要点：
- LangGraph 状态图实现多步骤有向工作流
- 状态管理与条件分支
- 循环执行（低质量回答自动重试）
- 多智能体协同（检索 Agent / 问答 Agent / 校验 Agent）
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, Field


# ===================== 状态定义 =====================

class ResearchState(BaseModel):
    """工作流全局状态"""

    # 输入
    question: str = ""
    task_type: Literal["qa", "summary"] = "qa"

    # 检索阶段
    retrieved_docs: list[Document] = Field(default_factory=list)
    reranked_docs: list[Document] = Field(default_factory=list)
    compressed_docs: list[Document] = Field(default_factory=list)

    # 生成阶段
    answer: str = ""
    summary: str = ""

    # 校验阶段
    validation_result: str = ""
    quality_score: float = 0.0
    retry_count: int = 0
    max_retries: int = 2

    # 元信息
    error: str = ""

    class Config:
        arbitrary_types_allowed = True


# ===================== 节点函数 =====================

async def retrieve_node(state: ResearchState) -> dict:
    """检索节点 - 混合检索（稠密向量 + BM25）"""
    logger.info(f"[检索节点] 查询: {state.question}")

    from app.rag.retriever import HybridRetriever
    from app.rag.vector_store import VectorStoreManager

    try:
        vsm = VectorStoreManager()
        hybrid = HybridRetriever(vector_store_manager=vsm)
        docs = await hybrid.retrieve(state.question, k=20)
        logger.info(f"[检索节点] 召回 {len(docs)} 个文档")
        return {"retrieved_docs": docs}
    except Exception as e:
        logger.error(f"[检索节点] 检索失败: {e}")
        return {"error": str(e)}


async def rerank_node(state: ResearchState) -> dict:
    """重排序节点 - BGE-reranker 精排"""
    logger.info(f"[重排节点] 对 {len(state.retrieved_docs)} 个文档重排")

    from app.rag.reranker import BGEReranker, ResultValidator

    reranker = BGEReranker(top_k=8)
    reranked = reranker.rerank(state.question, state.retrieved_docs)

    validator = ResultValidator()
    validated = validator.validate(reranked)

    logger.info(f"[重排节点] 重排后保留 {len(validated)} 个文档")
    return {"reranked_docs": validated}


async def compress_node(state: ResearchState) -> dict:
    """上下文压缩节点 - 提取与查询最相关的段落"""
    logger.info(f"[压缩节点] 压缩 {len(state.reranked_docs)} 个文档")

    from app.rag.reranker import ContextCompressor

    compressor = ContextCompressor()
    compressed = await compressor.compress(state.question, state.reranked_docs)

    return {"compressed_docs": compressed}


async def qa_node(state: ResearchState) -> dict:
    """学术问答节点 - 基于文献生成回答"""
    logger.info("[问答节点] 生成学术回答")

    from app.rag.chain import RAGChain

    chain = RAGChain()
    context_docs = state.compressed_docs or state.reranked_docs

    answer = await chain.answer(
        question=state.question,
        context_docs=context_docs,
        use_cot=True,
    )

    return {"answer": answer}


async def summary_node(state: ResearchState) -> dict:
    """摘要生成节点"""
    logger.info("[摘要节点] 生成文档摘要")

    from app.rag.chain import RAGChain

    chain = RAGChain()
    context_docs = state.compressed_docs or state.reranked_docs

    summary = await chain.summarize(context_docs)
    return {"summary": summary}


async def validate_node(state: ResearchState) -> dict:
    """结果校验节点 - 评估回答质量"""
    logger.info("[校验节点] 评估回答质量")

    from app.rag.chain import RAGChain

    chain = RAGChain()
    context_docs = state.compressed_docs or state.reranked_docs

    validation = await chain.validate_answer(
        question=state.question,
        context_docs=context_docs,
        answer=state.answer,
    )

    # 从校验结果中提取分数
    score = _extract_score(validation)

    return {
        "validation_result": validation,
        "quality_score": score,
        "retry_count": state.retry_count + 1,
    }


def _extract_score(validation_text: str) -> float:
    """从校验文本中提取质量分数"""
    patterns = [
        r"(\d+(?:\.\d+)?)\s*/\s*10",
        r"评分[：:]\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*分",
    ]
    for pattern in patterns:
        match = re.search(pattern, validation_text)
        if match:
            return float(match.group(1))
    return 5.0  # 默认中等分数


# ===================== 条件路由 =====================

def route_task_type(state: ResearchState) -> str:
    """根据任务类型路由到不同节点"""
    if state.error:
        return END
    return state.task_type


def route_after_validate(state: ResearchState) -> str:
    """校验后决定是否重试"""
    if state.quality_score >= 6.0:
        logger.info(f"[路由] 质量分数 {state.quality_score}，通过校验")
        return END
    if state.retry_count >= state.max_retries:
        logger.warning(f"[路由] 已达最大重试次数 {state.max_retries}")
        return END
    logger.info(
        f"[路由] 质量分数 {state.quality_score} < 6.0，"
        f"触发重试 ({state.retry_count}/{state.max_retries})"
    )
    return "retry"


# ===================== 构建工作流图 =====================

def build_research_workflow() -> StateGraph:
    """构建科研文档处理工作流

    流程图：
    START → 检索 → 重排 → 压缩 → [路由]
                                      ├→ qa → 校验 → [通过?]
                                      │                ├→ END (通过)
                                      │                └→ qa (重试)
                                      └→ summary → END
    """
    workflow = StateGraph(ResearchState)

    # 添加节点
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("compress", compress_node)
    workflow.add_node("qa", qa_node)
    workflow.add_node("summary", summary_node)
    workflow.add_node("validate", validate_node)

    # 添加边
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "compress")

    # 条件路由：任务类型
    workflow.add_conditional_edges(
        "compress",
        route_task_type,
        {
            "qa": "qa",
            "summary": "summary",
        },
    )

    # QA 后进入校验
    workflow.add_edge("qa", "validate")

    # 校验后条件路由：通过或重试
    workflow.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            END: END,
            "retry": "qa",
        },
    )

    # 摘要直接结束
    workflow.add_edge("summary", END)

    return workflow


def get_compiled_workflow():
    """获取编译后的工作流（可直接执行）"""
    workflow = build_research_workflow()
    return workflow.compile()
