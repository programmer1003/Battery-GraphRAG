import os
import json
import asyncio
from dotenv import load_dotenv
from datasets import Dataset

# ==========================================
# ⚙️ 1. 安全配置与黄金稳定版 Ragas 导入
# ==========================================
load_dotenv()
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-cb435f5fe94cac3a569e00b1c1872ce")

from ragas import evaluate
# 💥 修改点 1：导入大写类名，以便我们手动配置参数
from ragas.metrics import ContextRecall, Faithfulness, AnswerRelevancy

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from graph_rag_agent import agent_chat_stream, tool_query_kg

import logging
# 强制把 neo4j 的通知级别提高到 ERROR，屏蔽所有恶心的黄字 Warning
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

# ==========================================
# ⚙️ 2. 初始化裁判员 (直接使用 LangChain 原生对象)
# ==========================================
llm_judge = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=DEEPSEEK_KEY,
    openai_api_base="https://api.deepseek.com",
    max_retries=2
)

emb_judge = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-zh-v1.5")


# ==========================================
# 🧠 3. 评测数据管道搭建
# ==========================================
async def get_rag_response(question):
    context = await tool_query_kg(question)
    answer_text = ""
    async for event in agent_chat_stream(question):
        data = json.loads(event.replace("data: ", ""))
        if data["type"] == "final":
            answer_text = data["content"]
    return answer_text, [context]


async def prepare_dataset():
    with open("test_questions.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    questions, answers, contexts, truths = [], [], [], []

    for item in data:
        print(f"🔄 正在让系统作答: {item['question']}...")
        ans, ctx = await get_rag_response(item["question"])

        questions.append(item["question"])
        answers.append(ans)
        contexts.append(ctx)
        truths.append(item["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": truths
    })


# ==========================================
# 🎯 4. 主评测流程
# ==========================================
async def main():
    print("🚀 开始构建评测数据集...")
    ds = await prepare_dataset()

    print("\n⚖️ 正在交由 DeepSeek 与本地 BGE 联合打分，请稍候...")

    # 💥 修改点 2：强制 AnswerRelevancy 的 strictness=1，切断 n=3 的请求！
    cr = ContextRecall()
    f = Faithfulness()
    ar = AnswerRelevancy(strictness=1)

    results = evaluate(
        dataset=ds,
        metrics=[cr, f, ar],
        llm=llm_judge,
        embeddings=emb_judge
    )

    print("\n" + "=" * 50)
    print("🏆 电池专家系统 RAG 评测最终成绩单 🏆")
    print("=" * 50)
    print(results)

    results_df = results.to_pandas()
    results_df.to_csv("rag_evaluation_report.csv", index=False, encoding="utf-8-sig")
    print("\n✅ 详细评测报告已保存至 rag_evaluation_report.csv")


if __name__ == "__main__":
    asyncio.run(main())