# neo4j_client.py
import asyncio
from neo4j import AsyncGraphDatabase
import logging
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import logging
# 强制把 neo4j 的通知级别提高到 ERROR，屏蔽所有恶心的黄字 Warning
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


class AsyncNeo4jClient:
    """Neo4j 异步数据库单例客户端 (💥 工业级架构升级：原生 Vector Index 无状态版)"""

    def __init__(self, uri, user, password):
        try:
            self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password), max_connection_pool_size=50)
            logging.info("✅ 成功连接至 Neo4j 异步数据库")

            logging.info("🧠 正在加载 BGE 中文向量大模型，准备开启混合检索...")
            self.embedding_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

            # 💥 架构升级点：彻底抛弃了本地 node_cache 和 node_embeddings 缓存矩阵
            # 实现了 Agent 微服务节点的真正 "无状态化(Stateless)"
            logging.info("✅ 向量引擎挂载完毕，当前客户端处于无内存状态(Stateless)运行模式！")

        except Exception as e:
            logging.error(f"❌ 初始化失败: {e}")

    async def close(self):
        if self.driver:
            await self.driver.close()
            logging.info("🔌 数据库连接已安全关闭")

    async def hybrid_search_subgraph(self, semantic_query: str):
        """
        💥 核心 Hybrid RAG 逻辑：Top-K 算力下推！一次性召回前 3 个高优节点及其子图。
        """
        query_vector = self.embedding_model.encode(semantic_query).tolist()

        # 💥 核心修复：把 1 改成 3，扩大召回视野 (Top-K = 3)
        cypher_query = """
        CALL db.index.vector.queryNodes('entity_embeddings', 3, $query_vector)
        YIELD node AS anchor, score
        WHERE score >= 0.55
        OPTIONAL MATCH (anchor)-[r]-(neighbor)
        // 使用 WITH 聚合，按每个命中的核心节点打包它的邻居
        WITH anchor, score, collect(DISTINCT {relation: type(r), neighbor: neighbor.name}) AS context
        RETURN {
            matched_node: anchor.name,
            similarity_score: score,
            graph_context: context
        } AS result_dict
        """

        async with self.driver.session() as session:
            result = await session.run(cypher_query, query_vector=query_vector)
            # 💥 核心修复：不再只取 .single()，而是把前 3 个结果全部装进列表返回
            records = await result.data()

            if records:
                # 提取出所有的 result_dict 组成一个列表
                final_results = [record["result_dict"] for record in records if record["result_dict"]["matched_node"]]
                if final_results:
                    print(f"🎯 [Hybrid RAG] 视野扩大！成功命中 {len(final_results)} 个知识锚点！")
                    return final_results

            return None


# 👇 本地独立测试
async def main():
    load_dotenv()
    client = AsyncNeo4jClient(uri=os.getenv("NEO4J_URI"), user=os.getenv("NEO4J_USER"),
                              password=os.getenv("NEO4J_PASSWORD"))

    print("\n🔍 测试 1：混合检索 (无状态架构版)...")
    res = await client.hybrid_search_subgraph("电池内部好像长了小毛刺，导致漏电和压差变大了")
    print(f"\n✅ 混合检索召回结果:\n{res}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())