"""API 请求/响应数据模型"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ===================== 文档上传 =====================

class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    chunk_count: int = 0
    message: str = ""


class DocumentListItem(BaseModel):
    doc_id: str
    filename: str
    file_format: str
    status: str
    chunk_count: int
    created_at: datetime


# ===================== 问答 =====================

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: Optional[str] = Field(None, description="会话 ID")
    task_type: Literal["qa", "summary"] = Field("qa", description="任务类型")
    use_cache: bool = Field(True, description="是否使用缓存")


class SourceDocument(BaseModel):
    content: str
    filename: str
    page: Optional[int] = None
    section_title: Optional[str] = None
    relevance_score: Optional[float] = None


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceDocument] = []
    quality_score: Optional[float] = None
    validation: Optional[str] = None
    session_id: Optional[str] = None
    cached: bool = False


# ===================== 摘要 =====================

class SummaryResponse(BaseModel):
    summary: str
    source_count: int
    session_id: Optional[str] = None


# ===================== 通用 =====================

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    components: dict = {}
