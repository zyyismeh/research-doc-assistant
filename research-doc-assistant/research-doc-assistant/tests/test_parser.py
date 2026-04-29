"""文档解析模块单元测试"""

import tempfile
from pathlib import Path

import pytest

from app.document.parser import AcademicTextCleaner, DocumentParser


class TestAcademicTextCleaner:
    def test_clean_page_numbers(self):
        cleaner = AcademicTextCleaner()
        text = "这是正文内容 第 1 页 共 10 页 更多内容"
        result = cleaner.clean(text)
        assert "第" not in result or "页" not in result

    def test_clean_reference_brackets(self):
        cleaner = AcademicTextCleaner()
        text = "深度学习[1,2,3]在自然语言处理[4]中的应用"
        result = cleaner.clean(text)
        assert "[1,2,3]" not in result

    def test_extract_references(self):
        cleaner = AcademicTextCleaner()
        text = "[1] Author A. Title A. 2024. [2] Author B. Title B. 2023."
        refs = cleaner.extract_references(text)
        assert len(refs) == 2

    def test_segment_chinese(self):
        cleaner = AcademicTextCleaner()
        text = "深度学习在自然语言处理中的应用"
        result = cleaner.segment_chinese(text)
        assert " " in result  # 分词后有空格


class TestDocumentParser:
    def test_parse_markdown(self):
        parser = DocumentParser()
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
            f.write("# 标题\n\n这是正文内容\n\n## 子标题\n\n更多内容")
            f.flush()
            docs = parser.parse(f.name)
            assert len(docs) >= 1

    def test_parse_text(self):
        parser = DocumentParser()
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
            f.write("这是一段测试文本")
            f.flush()
            docs = parser.parse(f.name)
            assert len(docs) == 1
            assert "测试文本" in docs[0].page_content

    def test_unsupported_format(self):
        parser = DocumentParser()
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            with pytest.raises(ValueError, match="不支持"):
                parser.parse(f.name)
