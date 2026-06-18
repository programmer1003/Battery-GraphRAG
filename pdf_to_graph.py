# pdf_to_graph.py
import json
import asyncio
import sys
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from PyPDF2 import PdfReader
from openai import AsyncOpenAI
from neo4j_client import AsyncNeo4jClient
import os
from dotenv import load_dotenv

# 💥 架构补丁 1：引入本地轻量级向量模型
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 基础设施配置 (全异步化)
# ==========================================
load_dotenv()
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

# 实例化异步 LLM 客户端
client_llm = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 💥 架构补丁 2：全局加载向量模型，准备在入库时提取特征
print("🧠 正在加载 BGE 中文向量大模型，准备生成知识库特征...")
embedding_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# ==========================================
# 2. 核心功能函数 (你的原生逻辑 100% 保留)
# ==========================================

def extract_text_from_pdf(pdf_path: str, page_limit: int = 2) -> str:
    """PDF 文字提取器"""
    print(f"📄 正在解析 PDF: {pdf_path} ...")
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for i in range(min(page_limit, len(reader.pages))):
            text += reader.pages[i].extract_text()
        return text
    except Exception as e:
        print(f"❌ PDF 解析失败: {e}")
        return ""


async def llm_extract_triplets(text: str) -> list:
    """大模型化身无情的‘三元组抽取机器’ (带有 JSON 锁死功能)"""
    print("🧠 LLM 正在进行深度阅读理解与知识抽取，请稍候...")
    prompt = f"""
    你是一个工业级的知识图谱抽取专家。请阅读以下汽车维修手册文本，提取出所有的核心实体和关系。

    **支持的实体标签 (Label)**:
    - Fault (故障名称)
    - Symptom (症状表现)
    - RootCause (底层物理原因)
    - Solution (维修建议或改善措施)  <-- 💥 新增这个标签

    **支持的关系 (Relation)**:
    - HAS_SYMPTOM (连接 Fault 和 Symptom)
    - CAUSED_BY (连接 Fault 和 RootCause)
    - HAS_SOLUTION (连接 Fault 和 Solution) <-- 💥 新增这个关系

    **绝对指令**:
    请务必将提取的知识严格封装在 JSON 对象的 'triplets' 数组中返回。

    格式示例:
    {{
        "triplets": [
            {{"source_label": "Fault", "source_name": "电池热失控", "relation": "HAS_SYMPTOM", "target_label": "Symptom", "target_name": "冒白烟"}},
            {{"source_label": "Fault", "source_name": "电池热失控", "relation": "CAUSED_BY", "target_label": "RootCause", "target_name": "隔膜融化"}}
        ]
    }}

    待解析文本:
    {text}
    """
    try:
        response = await client_llm.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        raw_output = response.choices[0].message.content
        parsed_data = json.loads(raw_output)
        triplets = parsed_data.get("triplets", [])
        print(f"✅ 成功抽取 {len(triplets)} 条高质量知识三元组！")
        return triplets
    except Exception as e:
        print(f"❌ LLM 抽取或解析失败: {e}")
        return []

# 💥 架构补丁 3：自动在图数据库建立原生向量索引
async def init_vector_index(kg_client: AsyncNeo4jClient):
    print("⚙️ 正在初始化 Neo4j 底层原生向量索引...")
    # BGE-small 输出维度为 512，如果未来换模型需修改此处
    index_query = """
    CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
    FOR (e:Entity) ON (e.embedding)
    OPTIONS {indexConfig: {
      `vector.dimensions`: 512,
      `vector.similarity_function`: 'cosine'
    }}
    """
    async with kg_client.driver.session() as session:
        await session.run(index_query)


async def inject_triplets_batch(triplets: list, kg_client: AsyncNeo4jClient):
    """极速批量入库：带有原生向量特征持久化"""
    if not triplets:
        print("⚠️ 没有提取到知识，跳过入库。")
        return

    # 先拉起数据库索引
    await init_vector_index(kg_client)
    print("🕸️ 正在使用 UNWIND 技术批量打入云端图谱 (附带高维向量)...")

    # 💥 架构补丁 4：在存入数据库前，先算出每个实体的向量
    for item in triplets:
        # .tolist() 是为了将 numpy 数组转换为 Neo4j 可识别的标准格式
        item['source_embedding'] = embedding_model.encode(item['source_name']).tolist()
        item['target_embedding'] = embedding_model.encode(item['target_name']).tolist()

    # 原有的极度优雅的 UNWIND 逻辑保留，仅增加 a.embedding 和 b.embedding 的写入
    cypher = """
    UNWIND $batch_data AS item

    MERGE (a:Entity {name: item.source_name}) 
    ON CREATE SET a.type = item.source_label, a.embedding = item.source_embedding
    ON MATCH SET a.embedding = item.source_embedding

    MERGE (b:Entity {name: item.target_name}) 
    ON CREATE SET b.type = item.target_label, b.embedding = item.target_embedding
    ON MATCH SET b.embedding = item.target_embedding

    MERGE (a)-[r:RELATED_TO]->(b)
    ON CREATE SET r.type = item.relation
    """
    try:
        async with kg_client.driver.session() as session:
            await session.run(cypher, batch_data=triplets)
        print("🎉 批量知识(含特征向量)入库极速完成！")
    except Exception as e:
        print(f"❌ 批量入库失败: {e}")


# ==========================================
# 3. 异步引擎启动测试 (你的测试逻辑完全保留)
# ==========================================
# ==========================================
# 3. 异步引擎启动测试 (读取 TXT 弹药库)
# ==========================================
async def main():
    # 💥 修改点：改去读取我们刚刚创建的专业 txt 文件
    test_file_path = "mock_manual.txt"
    kg_client = AsyncNeo4jClient(uri=URI, user=USER, password=PASSWORD)

    try:
        # 读取本地 TXT 文件
        if os.path.exists(test_file_path):
            print(f"📄 正在读取文献: {test_file_path} ...")
            with open(test_file_path, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
        else:
            print("⚠️ 未找到 mock_manual.txt，使用备用极简文本...")
            extracted_text = "经过检测，发现电池组出现内阻增大故障，这通常会导致充电变慢的症状。底层原因是电极材料粉化。"

        # 呼叫 LLM 提炼知识 (它会自动阅读那一大长串专业文献)
        knowledge_triplets = await llm_extract_triplets(extracted_text)

        # 批量附带向量打入 Neo4j
        await inject_triplets_batch(knowledge_triplets, kg_client)

    finally:
        await kg_client.close()


if __name__ == "__main__":
    asyncio.run(main())