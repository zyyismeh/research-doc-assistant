"""文档管理 API - 上传、解析、向量化"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from loguru import logger

from app.core import settings
from app.document.chunker import AcademicChunker
from app.document.parser import DocumentParser, AcademicTextCleaner
from app.models.schemas import DocumentUploadResponse
from app.rag.vector_store import VectorStoreManager

router = APIRouter(prefix="/documents", tags=["文档管理"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传科研文档并自动解析、分块、向量化"""
    # 1. 校验文件
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    ext = Path(file.filename).suffix.lower()
    if ext not in DocumentParser.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，支持: {DocumentParser.SUPPORTED_EXTENSIONS}",
        )

    # 校验文件大小
    content = await file.read()
    max_size = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小超过限制: {settings.max_file_size_mb}MB",
        )

    # 2. 保存文件
    doc_id = str(uuid.uuid4())
    safe_filename = f"{doc_id}{ext}"
    file_path = settings.upload_path / safe_filename
    file_path.write_bytes(content)
    logger.info(f"文件已保存: {file_path}")

    try:
        # 3. 解析文档
        parser = DocumentParser()
        documents = parser.parse(file_path)

        # 4. 文本清洗
        cleaner = AcademicTextCleaner()
        for doc in documents:
            doc.page_content = cleaner.clean(doc.page_content)
            doc.metadata["doc_id"] = doc_id

        # 5. 语义分块
        chunker = AcademicChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.split_documents(documents)

        # 6. 向量化入库
        vsm = VectorStoreManager()
        await vsm.add_documents(chunks)

        return DocumentUploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            status="ready",
            chunk_count=len(chunks),
            message=f"文档解析完成，生成 {len(chunks)} 个知识块",
        )

    except Exception as e:
        logger.error(f"文档处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档"""
    upload_dir = settings.upload_path
    # 查找并删除文件
    for f in upload_dir.glob(f"{doc_id}.*"):
        f.unlink()
        logger.info(f"已删除文件: {f}")

    return {"message": f"文档 {doc_id} 已删除"}
