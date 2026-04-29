"""学术问答 API - 基于 LangGraph 工作流"""

from __future__ import annotations

import uuid

from fastapi import APIRouter
from loguru import logger

from app.db.cache import cache_manager
from app.models.schemas import (
    AnswerResponse,
    QuestionRequest,
    SourceDocument,
    SummaryResponse,
)
from app.workflow.graph import ResearchState, get_compiled_workflow

router = APIRouter(prefix="/chat", tags=["学术问答"])


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """学术问答 - 通过 LangGraph 工作流处理"""
    question = request.question.strip()
    session_id = request.session_id or str(uuid.uuid4())

    # 1. 检查缓存
    if request.use_cache:
        cached = await cache_manager.get_cached_answer(question)
        if cached:
            logger.info(f"命中缓存: {question[:50]}...")
            return AnswerResponse(
                answer=cached,
                session_id=session_id,
                cached=True,
            )

    # 2. 执行 LangGraph 工作流
    logger.info(f"执行工作流: task={request.task_type}, question={question[:50]}...")
    workflow = get_compiled_workflow()

    initial_state = ResearchState(
        question=question,
        task_type=request.task_type,
    )

    result = await workflow.ainvoke(initial_state)

    # 3. 处理结果
    if request.task_type == "summary":
        return AnswerResponse(
            answer=result.get("summary", ""),
            session_id=session_id,
        )

    # QA 结果
    answer = result.get("answer", "")
    quality_score = result.get("quality_score")
    validation = result.get("validation_result")

    # 构建来源文档
    sources = []
    for doc in result.get("reranked_docs", [])[:5]:
        sources.append(
            SourceDocument(
                content=doc.page_content[:300],
                filename=doc.metadata.get("filename", ""),
                page=doc.metadata.get("page"),
                section_title=doc.metadata.get("section_title"),
                relevance_score=doc.metadata.get("rerank_score"),
            )
        )

    # 4. 缓存结果
    if answer and request.use_cache:
        await cache_manager.cache_answer(question, answer)

    return AnswerResponse(
        answer=answer,
        sources=sources,
        quality_score=quality_score,
        validation=validation,
        session_id=session_id,
    )


@router.post("/summary", response_model=SummaryResponse)
async def generate_summary(request: QuestionRequest):
    """文档摘要生成"""
    request.task_type = "summary"
    session_id = request.session_id or str(uuid.uuid4())

    workflow = get_compiled_workflow()
    initial_state = ResearchState(
        question=request.question,
        task_type="summary",
    )

    result = await workflow.ainvoke(initial_state)

    return SummaryResponse(
        summary=result.get("summary", ""),
        source_count=len(result.get("reranked_docs", [])),
        session_id=session_id,
    )
