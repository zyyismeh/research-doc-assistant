# 🔬 智能科研文档助手

> 基于 LangChain + LangGraph + RAG 的科研文档智能问答系统

## 技术架构

```
┌──────────────────────────────────────────────────────┐
│                    FastAPI 服务层                      │
├──────────────────────────────────────────────────────┤
│                 LangGraph 工作流引擎                   │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  │
│  │ 检索  │→│ 重排  │→│ 压缩  │→│ 问答  │→│ 校验  │  │
│  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  │
├──────────────────────────────────────────────────────┤
│                   RAG 全栈                            │
│  稠密向量检索 + BM25 稀疏检索 → RRF 融合 → BGE 重排    │
├──────────────────────────────────────────────────────┤
│              文档处理 & 存储                           │
│  PDF/LaTeX/Word/MD → 语义分块 → Chroma/Milvus/FAISS  │
└──────────────────────────────────────────────────────┘
```

## 核心技术栈

| 模块 | 技术 |
|------|------|
| AI 框架 | LangChain, LangGraph, LangSmith |
| RAG | 混合检索, 语义分块, BGE-reranker, HyDE, 上下文压缩 |
| 大模型 | 通义千问/GPT/Ollama, BGE/m3e 嵌入 |
| 文档处理 | PyMuPDF, pdfplumber, python-docx, pylatexenc |
| 向量数据库 | Chroma, FAISS, Milvus |
| 后端 | FastAPI, 异步编程 |
| 存储 | PostgreSQL, Redis |
| 部署 | Docker, Docker Compose |

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url>
cd research-doc-assistant

# 复制环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key
```

### 2. 本地开发

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -e ".[dev]"

# 启动服务
uvicorn app.main:app --reload --port 8000
```

### 3. Docker 部署

```bash
# 一键启动（应用 + PostgreSQL + Redis）
docker-compose up -d

# 查看日志
docker-compose logs -f app
```

### 4. 使用 Ollama 本地模型（可选）

```bash
# 安装 Ollama 并拉取模型
ollama pull qwen2.5:7b

# .env 配置
LLM_PROVIDER=ollama
LLM_MODEL_NAME=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
```

## API 接口

### 健康检查
```bash
curl http://localhost:8000/health
```

### 上传文档
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@paper.pdf"
```

### 学术问答
```bash
curl -X POST http://localhost:8000/api/v1/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "这篇论文的主要研究方法是什么？"}'
```

### 生成摘要
```bash
curl -X POST http://localhost:8000/api/v1/chat/summary \
  -H "Content-Type: application/json" \
  -d '{"question": "请总结这篇论文的核心贡献"}'
```

### 批量导入文档
```bash
python scripts/ingest.py ./data/uploads
```

## 项目结构

```
research-doc-assistant/
├── app/
│   ├── core/              # 核心配置、LLM 工厂
│   │   ├── __init__.py    # Settings 配置类
│   │   ├── llm_factory.py # 大模型与嵌入模型工厂
│   │   └── logging.py     # 日志配置
│   ├── document/          # 文档处理模块
│   │   ├── parser.py      # 多格式文档解析器
│   │   └── chunker.py     # 语义分块 + HyDE
│   ├── rag/               # RAG 核心模块
│   │   ├── vector_store.py # 向量数据库管理
│   │   ├── retriever.py   # 混合检索（向量+BM25+RRF）
│   │   ├── reranker.py    # BGE 重排 + 上下文压缩
│   │   └── chain.py       # Prompt 工程 + RAG 链路
│   ├── workflow/           # LangGraph 工作流
│   │   └── graph.py       # 多步骤工作流（检索→重排→问答→校验）
│   ├── api/               # FastAPI 路由
│   │   ├── documents.py   # 文档上传管理
│   │   └── chat.py        # 学术问答
│   ├── db/                # 数据库
│   │   ├── models.py      # SQLAlchemy 模型
│   │   └── cache.py       # Redis 缓存管理
│   ├── models/            # Pydantic 数据模型
│   │   └── schemas.py     # 请求/响应 Schema
│   └── main.py            # FastAPI 应用入口
├── tests/                 # 单元测试
├── scripts/               # 工具脚本
│   └── ingest.py          # 批量文档导入
├── data/
│   ├── uploads/           # 上传文件存储
│   └── vectorstore/       # 向量数据库持久化
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── README.md
```

## LangGraph 工作流说明

```
START → 检索(混合检索) → 重排(BGE-reranker) → 压缩(上下文压缩)
                                                      │
                                          ┌───────────┤
                                          ▼           ▼
                                     [QA 问答]    [摘要生成] → END
                                          │
                                          ▼
                                     [质量校验]
                                          │
                                    ┌─────┤
                                    ▼     ▼
                                  END   [重试QA]（分数<6 且 未超重试次数）
```

## License

MIT
