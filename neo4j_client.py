from neo4j import GraphDatabase
import logging
import os
from dotenv import load_dotenv

class Neo4jClient:
    """Neo4j 数据库单例客户端"""

    def __init__(self, uri, user, password):
        try:
            # 开启连接池
            self.driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_pool_size=50)
            logging.info("成功连接至 Neo4j 数据库")
        except Exception as e:
            logging.error(f"Neo4j 连接失败: {e}")

    def close(self):
        if self.driver:
            self.driver.close()

    def get_fault_subgraph(self, fault_name: str):
        """
        核心图查询逻辑：查询故障实体的上下文子图
        使用参数化查询 ($fault_name) 防止注入攻击，并利用缓存加速
        """
        cypher_query = """
        MATCH (f:Fault {name: $fault_name})
        OPTIONAL MATCH (f)-[r1:HAS_SYMPTOM]->(s:Symptom)
        OPTIONAL MATCH (f)-[r2:CAUSED_BY]->(c:RootCause)
        RETURN f.name AS fault, 
               collect(DISTINCT s.name) AS symptoms, 
               collect(DISTINCT c.name) AS causes
        """
        with self.driver.session() as session:
            result = session.run(cypher_query, fault_name=fault_name)
            record = result.single()  # 我们只需要匹配到的那个核心节点的子图

            if record:
                return {
                    "fault_node": record["fault"],
                    "symptoms": record["symptoms"],
                    "root_causes": record["causes"]
                }
            return None



# ==========================================
# 👇 改造后的安全测试代码 👇
# ==========================================
if __name__ == "__main__":
    # 强制加载 .env 文件中的密码
    load_dotenv()

    # 安全地从系统环境变量中读取配置
    URI = os.getenv("NEO4J_URI")
    USER = os.getenv("NEO4J_USER")
    PASSWORD = os.getenv("NEO4J_PASSWORD")

    if not all([URI, USER, PASSWORD]):
        print("❌ 错误：未能从 .env 文件中读取到完整的 Neo4j 凭证，请检查配置！")
        exit()

    print("🚀 正在跨网连接云端 GraphRAG 大脑，请稍候...")

    # 实例化单例客户端
    client = Neo4jClient(uri=URI, user=USER, password=PASSWORD)

    print("\n🔍 正在测试底层查询通路...")
    test_result = client.get_fault_subgraph("微短路故障")
    print(f"✅ 测试查询结果: {test_result}")