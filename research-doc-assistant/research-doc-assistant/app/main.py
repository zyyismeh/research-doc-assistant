"""FastAPI 主应用入口"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core import settings
from app.models.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("🚀 智能科研文档助手启动中...")
    logger.info(f"   LLM 提供商: {settings.llm_provider}")
    logger.info(f"   向量库类型: {settings.vector_store_type}")
    logger.info(f"   嵌入模型:   {settings.embedding_model_name}")

    # 初始化数据库（如果可用）
    try:
        from app.db.models import init_db
        await init_db()
        logger.info("   PostgreSQL 初始化完成")
    except Exception as e:
        logger.warning(f"   PostgreSQL 初始化跳过: {e}")

    yield

    # 清理资源
    from app.db.cache import cache_manager
    await cache_manager.close()
    logger.info("应用已关闭")


app = FastAPI(
    title="智能科研文档助手",
    description="基于 LangChain + LangGraph + RAG 的科研文档智能问答系统",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
from app.api.documents import router as documents_router
from app.api.chat import router as chat_router

app.include_router(documents_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    components = {
        "llm": settings.llm_provider,
        "vector_store": settings.vector_store_type,
        "embedding": settings.embedding_model_name,
    }

    # 检查 Redis
    try:
        from app.db.cache import cache_manager
        await cache_manager.client.ping()
        components["redis"] = "connected"
    except Exception:
        components["redis"] = "unavailable"

    return HealthResponse(status="ok", components=components)
