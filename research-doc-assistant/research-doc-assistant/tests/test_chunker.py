"""分块模块单元测试"""

from langchain_core.documents import Document

from app.document.chunker import AcademicChunker


class TestAcademicChunker:
    def test_basic_chunking(self):
        chunker = AcademicChunker(chunk_size=100, chunk_overlap=10)
        docs = [
            Document(
                page_content="这是一段很长的文本。" * 50,
                metadata={"source": "test.pdf"},
            )
        ]
        chunks = chunker.split_documents(docs)
        assert len(chunks) > 1
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_table_not_chunked(self):
        chunker = AcademicChunker(chunk_size=100, chunk_overlap=10)
        docs = [
            Document(
                page_content="| A | B |\n|---|---|\n| 1 | 2 |",
                metadata={"content_type": "table", "source": "test.pdf"},
            )
        ]
        chunks = chunker.split_documents(docs)
        assert len(chunks) == 1

    def test_formula_not_chunked(self):
        chunker = AcademicChunker(chunk_size=100, chunk_overlap=10)
        docs = [
            Document(
                page_content="E = mc^2",
                metadata={"content_type": "formula", "source": "test.pdf"},
            )
        ]
        chunks = chunker.split_documents(docs)
        assert len(chunks) == 1
