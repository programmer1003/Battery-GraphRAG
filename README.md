# 🔋 动力电池故障诊断 Agent：基于知识图谱的 GraphRAG 系统

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)
![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-008CC1.svg)
![LLM](https://img.shields.io/badge/LLM-DeepSeek_API-brightgreen.svg)
![Ragas](https://img.shields.io/badge/Eval-Ragas-FF5A5F.svg)

**🚀 工业级智能体诊断引擎**：面向新能源车企售后排障场景，结合 Agent 工具调用、时序异常检测模型（TF-GDC）与知识图谱底座，解决大模型在垂直领域“多跳推理缺失”、“多模态数据阻塞”及“诊断幻觉”等工程痛点。

## 📖 项目简介

本项目摒弃了传统的单纯向量检索（Vector RAG），构建了一套 **GraphRAG + Function Calling** 架构的多端微服务系统。
系统不仅能从维修文献中自动构建图谱，还能通过 Agent 动态跨服务调度 GPU 算力节点，实现“文本诊断语义”与“底层三维物理波形（电压/电流/温度）”的实时联合会诊与流式渲染。

## ✨ 核心工程特性

- **🕸️ GraphRAG 图谱构建与算力下推**：通过 `pdf_to_graph.py` 管线，利用 LLM 提取 [故障-症状-根因-措施] 强约束三元组并自动建图。集成 `bge-small-zh-v1.5` 将特征持久化至 Neo4j，利用底层引擎执行 KNN 原生检索，实现应用层无状态（Stateless）并发。
- **🧠 Agent 路由与多模态联合诊断**：基于 DeepSeek Function Calling 搭建中枢，内置 Query Rewrite 将车间口语对齐标准术语。支持跨微服务调用 TF-GDC 时序异常检测算法，联合评估底层重构误差与图谱物理逻辑。
- **⚡ 前后端解耦与 SSE 流式交互**：针对算法节点返回的海量物理波形数组耗尽 Token 的问题，实施数据分流机制。后端将时序数组剥离，通过 Server-Sent Events (SSE) 协议向前端独立推流，实现 Agent 文本思考与 ECharts 物理波形图的异步同步渲染。
- **📊 Ragas 量化评测与幻觉抑制**：搭建端到端量化评测流水线 `evaluate_rag.py`。针对 LLM-as-a-judge 判定 JSON 上下文时的“水床效应”，通过精调图谱召回阈值（Top-K=3）与注入强约束反思 Prompt，将系统答案相关性（Relevancy）与召回率稳定在 70%+ 可用基线。

## 🏗️ 系统架构与微服务矩阵

系统采用解耦的微服务架构，包含 3 个独立运行的服务节点：

```text
[UI 交互层: Port 8501] ◀──(SSE 流式推流: 文本 + V/I/T 波形)──▶ [Agent 网关层: Port 8000]
 (app.py)                                                  (api_main.py)
                                                                 │
                                                                 ├──▶ (Function 1) ──▶ [Neo4j Aura 云端图谱]
                                                                 │                     (KNN 特征匹配)
                                                                 │
                                                                 └──▶ (Function 2) ──▶ [算法算力层: Port 8001]
                                                                                       (model_server.py / TF-GDC)

---
## 🛠️ 技术栈 (Tech Stack)
- ** AI & Agent：DeepSeek Chat API, GraphRAG, Function Calling, Prompt Engineering
- ** 图谱与向量引擎：Neo4j AsyncGraphDatabase, BAAI/bge-small-zh-v1.5数据工程与图谱：Neo4j AuraDB (Cypher原生检索), PyPDF2, BGE Embedding
- ** 时序检测算法：PyTorch (Transformer-based TF-GDC)
- ** 后端与通信：FastAPI, Uvicorn, Server-Sent Events (SSE), HTTPX前端与部署：Streamlit, Docker, Docker Compose
- ** 前端与评测：Streamlit, Ragas, Datasets

## 🚀 快速开始 (Quick Start)
**1.配置环境变量**
配置环境变量
在根目录创建 .env 文件，并填入你的鉴权信息：

```bash
NEO4J_URI=neo4j+s://您的数据库地址.databases.neo4j.io
NEO4J_USER=您的数据库账号
NEO4J_PASSWORD=您的数据库密码
DEEPSEEK_API_KEY=sk-您的真实秘钥
```
**2. 知识库自动化构建**
执行图谱初始化脚本，解析 mock_manual.txt 等文献，生成高维特征并写入 Neo4j 云端：
```bash
python pdf_to_graph.py
```

**3. 启动微服务集群 (需开启 3 个终端)**
-** 终端 1: 启动底层 TF-GDC 算法算力节点
```bash
python model_server.py  # 挂载于 8001 端口
```
-** 终端 2: 启动 Agent 诊断大脑网关
```bash
python api_main.py      # 挂载于 8000 端口
```

-** 终端 3: 启动交互式可视化 UI
```bash
streamlit run app.py    # 挂载于 8501 端口
```

**4. 运行自动化评测**
-** 执行 Ragas 测试脚本，读取 test_questions.json 对当前图谱上下文进行量化打分：
```bash
python evaluate_rag.py
```

### 方式一：Docker 极速启动（推荐）
确保你的机器上已安装 Docker 和 docker-compose。

**1.克隆仓库**
```bash
git clone [https://github.com/你的用户名/Battery-GraphRAG.git](https://github.com/你的用户名/Battery-GraphRAG.git)
cd Battery-GraphRAG
```


```bash
docker-compose up --build -d
前端访问地址：http://localhost:8501
FastAPI 接口文档：http://localhost:8000/docs
```

###  方式二：本地开发环境启动

```bash
安装依赖：pip install -r requirements.txt
启动 FastAPI 后端：python api_main.py
另起一个终端，启动前端：streamlit run app.py
```

## 踩坑与优化记录 (Troubleshooting)
- ** 在工程化落地过程中，本系统解决并沉淀了以下关键技术问题：
- ** 大规模特征比对导致的 OOM 优化：初期在本地计算 Cosine Similarity 时极易导致内存溢出。最终通过将 Embedding 持久化，并将向量检索任务下推至 Neo4j 底层引擎（db.index.vector.queryNodes），实现毫秒级召回并彻底解决 OOM 问题。
- ** 长数组导致的大模型推理阻塞：多模态诊断中，直接将高频采样数据输入 LLM 会导致 Token 超限或响应严重延迟。采用“计算与渲染分离”策略，后端解析出物理路径后直接通过 SSE 推送给前端渲染 ECharts 曲线，LLM 仅负责解读结论。
**📂 项目目录结构 (Project Structure)**
Battery-GraphRAG/
```bash
├── api_main.py           # [Port 8000] FastAPI 网关入口，封装 SSE 路由
├── app.py                # [Port 8501] Streamlit 前端，支持波形渲染与打字机特效
├── model_server.py       # [Port 8001] 独立算法服务，提供 TF_GDC 模型推理与 CSV 时序加载
├── graph_rag_agent.py    # Agent 核心逻辑：包含 Tool 定义、跨服务调度与极严苛系统 Prompt
├── neo4j_client.py       # 无状态化的 Neo4j 异步驱动，实现 Hybrid RAG 与算力下推
├── pdf_to_graph.py       # 自动化 Text2Graph 建图管线 (含 BGE 向量生成)
├── evaluate_rag.py       # 基于 Ragas 的端到端量化评测流水线
├── dataset/              # 存放电池原始 CSV 时序数据 (如 B008.csv)
├── test_questions.json   # 评测基准题库 (Ground Truth)
└── mock_manual.txt       # 用于模拟初始化图谱的知识文献
```