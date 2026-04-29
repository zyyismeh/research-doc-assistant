"""文档入库脚本 - 批量导入科研文档"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from app.document.chunker import AcademicChunker
from app.document.parser import AcademicTextCleaner, DocumentParser
from app.rag.vector_store import VectorStoreManager


async def ingest_directory(directory: str):
    """批量导入目录下的所有文档"""
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.error(f"目录不存在: {directory}")
        return

    parser = DocumentParser()
    cleaner = AcademicTextCleaner()
    chunker = AcademicChunker(chunk_size=512, chunk_overlap=64)
    vsm = VectorStoreManager()

    files = [
        f
        for f in dir_path.iterdir()
        if f.suffix.lower() in parser.SUPPORTED_EXTENSIONS
    ]
    logger.info(f"发现 {len(files)} 个待处理文档")

    total_chunks = 0
    for file_path in files:
        try:
            logger.info(f"处理文件: {file_path.name}")
            docs = parser.parse(file_path)

            for doc in docs:
                doc.page_content = cleaner.clean(doc.page_content)

            chunks = chunker.split_documents(docs)
            await vsm.add_documents(chunks)
            total_chunks += len(chunks)
            logger.info(f"  ✓ {file_path.name}: {len(chunks)} 个块")
        except Exception as e:
            logger.error(f"  ✗ {file_path.name}: {e}")

    logger.info(f"批量导入完成: {len(files)} 个文件, {total_chunks} 个块")


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "./data/uploads"
    asyncio.run(ingest_directory(directory))
