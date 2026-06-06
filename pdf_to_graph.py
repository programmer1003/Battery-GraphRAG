import json
from PyPDF2 import PdfReader
from openai import OpenAI
from neo4j_client import Neo4jClient
import os
from dotenv import load_dotenv

# 自动读取 .env 文件里的密码
load_dotenv()
# ==========================================
# 1. 基础设施配置
# ==========================================
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

client_llm = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

kg_client = Neo4jClient(uri=URI, user=USER, password=PASSWORD)


# ==========================================
# 2. 核心功能函数
# ==========================================

def extract_text_from_pdf(pdf_path: str, page_limit: int = 2) -> str:
    """极其暴力的 PDF 文字提取器 (为了测试，默认只读前2页防超时)"""
    print(f"📄 正在解析 PDF: {pdf_path} ...")
    reader = PdfReader(pdf_path)
    text = ""
    # 工业级防呆：防止 PDF 太长把大模型撑爆，限定页数
    for i in range(min(page_limit, len(reader.pages))):
        text += reader.pages[i].extract_text()
    return text


def llm_extract_triplets(text: str) -> list:
    """大模型化身无情的‘三元组抽取机器’"""
    print("🧠 LLM 正在进行深度阅读理解与知识抽取，请稍候...")

    # 🌟 简历绝杀级 Prompt Engineering (强制 JSON 输出)
    prompt = f"""
    你是一个工业级的知识图谱抽取专家。请阅读以下汽车维修手册文本，提取出所有的核心实体和关系。

    **支持的实体标签 (Label)**:
    - Fault (故障名称)
    - Symptom (症状表现)
    - RootCause (底层物理原因)

    **支持的关系 (Relation)**:
    - HAS_SYMPTOM (连接 Fault 和 Symptom)
    - CAUSED_BY (连接 Fault 和 RootCause)

    **绝对指令**:
    请务必**严格且仅输出**合法的 JSON 数组，不要输出任何 Markdown 标记（如 ```json），不要包含任何解释性废话！

    格式示例:
    [
        {{"source_label": "Fault", "source_name": "电池热失控", "relation": "HAS_SYMPTOM", "target_label": "Symptom", "target_name": "冒白烟"}},
        {{"source_label": "Fault", "source_name": "电池热失控", "relation": "CAUSED_BY", "target_label": "RootCause", "target_name": "隔膜融化"}}
    ]

    待解析文本:
    {text}
    """

    response = client_llm.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1  # 温度调极低，防止大模型产生创造力导致 JSON 格式崩溃
    )

    raw_output = response.choices[0].message.content.strip()

    try:
        # 清理可能存在的 markdown 标记
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:-3]
        triplets = json.loads(raw_output)
        print(f"✅ 成功抽取 {len(triplets)} 条知识三元组！")
        return triplets
    except Exception as e:
        print(f"❌ LLM 输出格式破坏，解析 JSON 失败: {raw_output}")
        return []


def inject_triplets_to_neo4j(triplets: list):
    """自动将 JSON 转换为 Cypher 并打入云端图谱"""
    if not triplets:
        return

    print("🕸️ 正在将提取的知识注入 Neo4j 云端大脑...")

    with kg_client.driver.session() as session:
        for item in triplets:
            src_label = item['source_label']
            src_name = item['source_name']
            rel = item['relation']
            tgt_label = item['target_label']
            tgt_name = item['target_name']

            # 动态拼接极其强大的 MERGE 语句
            cypher = f"""
            MERGE (a:{src_label} {{name: $src_name}})
            MERGE (b:{tgt_label} {{name: $tgt_name}})
            MERGE (a)-[r:{rel}]->(b)
            """
            session.run(cypher, src_name=src_name, tgt_name=tgt_name)
            print(f"   -> 写入连接: [{src_name}] -({rel})-> [{tgt_name}]")

    print("🎉 自动化知识入库全链路完成！")


# ==========================================
# 3. 引擎启动
# ==========================================
if __name__ == "__main__":
    # ⚠️ 请在电脑桌面上随便准备一个关于电池维修的 PDF，把路径填在这里
    # 如果是 Windows，注意路径里用双斜杠 \\ 或者在前面加 r
    test_pdf_path = r"C:\Users\赵旭东\Desktop\Battery_Agent\Battery_RAG\battery_knowledge.pdf"

    try:
        # Step 1: 撕开 PDF
        extracted_text = extract_text_from_pdf(test_pdf_path, page_limit=1)

        # Step 2: 呼叫 LLM 提炼知识
        knowledge_triplets = llm_extract_triplets(extracted_text)

        # Step 3: 打入 Neo4j
        inject_triplets_to_neo4j(knowledge_triplets)

    except FileNotFoundError:
        print(f"❌ 找不到 PDF 文件: {test_pdf_path}，请检查路径。")