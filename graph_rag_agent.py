import json
from openai import OpenAI
from neo4j_client import Neo4jClient
import os
from dotenv import load_dotenv

# 自动读取 .env 文件里的密码
load_dotenv()
# ==========================================
# 1. 你的配置 (保持你的真实 Key 不变)
# ==========================================
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

client_llm = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # 💥 你的真实 Key
    base_url="https://api.deepseek.com"
)

kg_client = Neo4jClient(uri=URI, user=USER, password=PASSWORD)


# ==========================================
# 🌟 新增：数据库动态感知探针
# ==========================================
def get_dynamic_fault_list() -> list:
    """每次对话前，先去数据库里摸一次底，看看现在到底学会了哪些故障"""
    query = "MATCH (f:Fault) RETURN collect(DISTINCT f.name) AS names"
    try:
        with kg_client.driver.session() as session:
            result = session.run(query).single()
            # 如果数据库是空的，返回空列表；否则返回真实的故障名称列表
            return result["names"] if result and result["names"] else []
    except Exception as e:
        print(f"⚠️ 动态感知探针异常: {e}")
        return []


# ==========================================
# 2. 核心问答引擎
# ==========================================
def graph_rag_chat(user_question: str):
    print(f"\n👤 用户提问: {user_question}")
    print("--------------------------------------------------")

    # --------------------------------------------------
    # 步骤 1：动态意图识别 (Dynamic Text2Entity)
    # --------------------------------------------------
    print("🤖 Agent 思考中：正在启动动态感知并分析意图...")

    # 💥 核心：在这里调用探针，获取最新知识库名单
    known_faults = get_dynamic_fault_list()

    extraction_prompt = f"""
    你是一个智能的意图识别专家。用户的输入是："{user_question}"

    目前我们的Neo4j图数据库中，**真正存在**的故障实体名单如下：
    {known_faults}

    请执行以下逻辑：
    1. 分析用户的输入，看看他想问的故障，是否在上述的名单中（允许语义相近的匹配）。
    2. 如果匹配到了名单中的某个故障，请**只输出那个具体的故障名称**。
    3. 如果用户问的故障完全不在名单中，或者问题太模糊，请严格输出 "None"。
    4. 不要输出任何标点符号和解释性废话。
    """

    response = client_llm.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": extraction_prompt}],
        temperature=0.0
    )

    fault_keyword = response.choices[0].message.content.strip()

    # 🛡️ 极其优雅的防呆拦截
    if fault_keyword == "None" or fault_keyword not in known_faults:
        print("❌ 拦截：用户问题超纲或未命中图谱节点")
        return f"抱歉，关于您提到的问题，我目前的图谱知识库中还没有相关的数据。我现在掌握的故障知识包括：{', '.join(known_faults[:5])}等。您可以换个问题，或者上传新的维修手册让我学习。"

    print(f"🎯 成功命中图谱实体: [{fault_keyword}]")

    # --------------------------------------------------
    # 步骤 2：拿着实体去 Neo4j 捞数据 (Graph Retrieval)
    # --------------------------------------------------
    print("🕸️ Agent 行动中：正在从云端 Neo4j 抓取多跳关联知识...")
    subgraph_data = kg_client.get_fault_subgraph(fault_keyword)

    if not subgraph_data:
        return "数据库节点异常，未找到关联信息。"

    print(f"📦 成功捞出图谱子图: {subgraph_data}")

    # --------------------------------------------------
    # 步骤 3：图文融合，生成最终专家的回答 (RAG Generation)
    # --------------------------------------------------
    print("✨ Agent 总结中：正在融合图谱数据与大模型语义...")
    generation_prompt = f"""
    你是一位资深的新能源汽车电池维修专家。
    用户的问题是："{user_question}"

    我在图数据库中为你查询到了以下极其精确的专家知识：
    - 故障名称：{subgraph_data['fault_node']}
    - 可能会出现的症状：{', '.join(subgraph_data['symptoms'])}
    - 导致该故障的底层根因：{', '.join(subgraph_data['root_causes'])}

    请结合上述知识，用专业、耐心、且容易听懂的口吻回答用户。
    注意：必须严格基于我提供的知识来回答，不能自己瞎编根因！
    """

    final_response = client_llm.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": generation_prompt}],
        temperature=0.5
    )

    final_answer = final_response.choices[0].message.content
    print("==================================================")
    print(f"👩‍🔧 最终诊断专家回复:\n\n{final_answer}")
    print("==================================================")
    return final_answer