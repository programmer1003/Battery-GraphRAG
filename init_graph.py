from neo4j_client import Neo4jClient
import os
from dotenv import load_dotenv

# 自动读取 .env 文件里的密码
load_dotenv()
# ==========================================
# 1. 填入你的大神三件套
# ==========================================
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")


# ==========================================
# 2. 编写硬核的知识注入脚本 (Cypher 语句)
# ==========================================
def inject_knowledge_to_cloud(client):
    """把电池专家的知识，强行打入云端图谱大脑"""

    # 这是一段极其暴力的 Cypher 插入语句（MERGE表示如果不存在就创建）
    cypher_insert = """
    // 1. 创建核心故障节点
    MERGE (f1:Fault {name: "微短路故障"})
    MERGE (f2:Fault {name: "老化波动"})

    // 2. 创建症状节点
    MERGE (s1:Symptom {name: "压差逐渐增大"})
    MERGE (s2:Symptom {name: "温度局部异常升高"})
    MERGE (s3:Symptom {name: "容量衰减"})

    // 3. 创建根因节点
    MERGE (c1:RootCause {name: "锂枝晶刺穿隔膜"})
    MERGE (c2:RootCause {name: "SEI膜不断增厚"})

    // 4. 建立神级关联 (拉红线！)
    MERGE (f1)-[:HAS_SYMPTOM]->(s1)
    MERGE (f1)-[:HAS_SYMPTOM]->(s2)
    MERGE (f1)-[:CAUSED_BY]->(c1)

    MERGE (f2)-[:HAS_SYMPTOM]->(s3)
    MERGE (f2)-[:CAUSED_BY]->(c2)
    """

    print("🧠 正在向云端数据库注射专家知识...")
    # 调用底层驱动直接执行写入
    with client.driver.session() as session:
        session.run(cypher_insert)
    print("✅ 知识注入完毕！图谱神经元已连接！")


if __name__ == "__main__":
    # 连接数据库
    client = Neo4jClient(uri=URI, user=USER, password=PASSWORD)

    # 执行注入
    inject_knowledge_to_cloud(client)

    # 再次查询测试（见证奇迹的时刻）
    print("\n🔍 正在验证多跳查询...")
    result = client.get_fault_subgraph("微短路故障")

    print("\n🎉 最终召回结果：")
    print(result)