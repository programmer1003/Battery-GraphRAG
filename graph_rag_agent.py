# graph_rag_agent.py
import json
import asyncio
import sys
from openai import AsyncOpenAI
from neo4j_client import AsyncNeo4jClient
import os
from dotenv import load_dotenv
import httpx  # 👈 导入 HTTPX

# 解决 Windows 异步报错
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

# 实例化异步组件
client_llm = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
kg_client = AsyncNeo4jClient(uri=URI, user=USER, password=PASSWORD)


# ==========================================
# 🛠️ 核心组件：Agent 的工具箱 (Tools)
# ==========================================

async def tool_query_kg(semantic_query: str) -> str:
    """工具 1：调用底层的 Hybrid RAG (🔥 已集成 Query Rewrite)"""
    print(f"\n   🧠 [Query Rewrite] 正在拦截并翻译口语化查询...")

    rewrite_prompt = f"""
    你是一个新能源电池领域的资深专家。用户的原始查询是："{semantic_query}"
    请将里面口语化的描述翻译成标准的电池物理与电化学专业术语。
    - 例如：“小毛刺” -> “锂枝晶 隔膜刺穿”
    - 例如：“鼓包” -> “产气膨胀”
    - 例如：“充不进电” -> “析锂 内阻增大”
    如果原始查询已经很专业，请提取出核心名词即可。
    【强制输出格式】：只输出重写后的标准关键词（用空格隔开），绝对不要输出任何多余的解释、标点或句子！
    """

    try:
        rewrite_response = await client_llm.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.0
        )
        standard_query = rewrite_response.choices[0].message.content.strip()
        print(f"   ✨ [Query Rewrite] 翻译成功: '{semantic_query}' -> '{standard_query}'")
    except Exception as e:
        print(f"   ⚠️ [Query Rewrite] 翻译失败，降级使用原查询。原因: {e}")
        standard_query = semantic_query

    print(f"   🔧 [Tool 执行] 启动混合检索引擎匹配标准语义: '{standard_query}'...")
    result = await kg_client.hybrid_search_subgraph(standard_query)

    if not result:
        return json.dumps({"error": f"图谱中未找到与 {standard_query} 相关的内容。"})

    return json.dumps(result, ensure_ascii=False)


async def tool_analyze_time_series(battery_id: str) -> str:
    """工具 2：通过 HTTP 真实呼叫部署在 8001 端口的 TF-GDC 算法引擎"""
    print(f"\n   📈 [Tool 执行] 正在跨服务呼叫 GPU/CPU 算力节点的 TF-GDC 引擎分析 {battery_id}...")

    try:
        # 发起异步 HTTP 请求到你的模型微服务，防代理劫持
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.post(
                "http://127.0.0.1:8001/predict",
                json={"battery_id": battery_id},
                timeout=15.0  # 给 PyTorch 留 15 秒的推理时间
            )

        if response.status_code == 200:
            result_json = response.json()
            return json.dumps(result_json, ensure_ascii=False)
        else:
            return json.dumps({"error": f"算法引擎返回错误状态码: {response.status_code}"})

    except Exception as e:
        return json.dumps({"error": f"无法连接到算法引擎，请检查 8001 端口是否启动: {e}"})


# ==========================================
# 🧠 核心组件：Agent 大脑 (Tool-Calling 路由)
# ==========================================

tools_definition = [
    {
        "type": "function",
        "function": {
            "name": "tool_query_kg",
            "description": "维修知识库混合检索工具。无论用户输入多么口语化的症状描述、报警信息或模糊的故障原因，都可以直接传入，底层向量引擎会自动匹配专业知识。",
            "parameters": {
                "type": "object",
                "properties": {
                    "semantic_query": {"type": "string", "description": "用户的模糊描述、疑似故障名或报警信息"}},
                "required": ["semantic_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_analyze_time_series",
            "description": "电池实时状态诊断工具。当用户提供具体的电池编号(如 B008)，要求检查当前状态或预测风险时调用。",
            "parameters": {
                "type": "object",
                "properties": {"battery_id": {"type": "string", "description": "电池的编号"}},
                "required": ["battery_id"]
            }
        }
    }
]


# ==========================================
# 🧠 核心组件：Agent 大脑 (多轮 ReAct + 反思增强版)
# ==========================================
async def agent_chat_stream(user_question: str):
    print(f"\n👤 用户提问: {user_question}")
    print("🤖 Agent 启动深度思考引擎...")

    def make_event(msg_type, content):
        return f"data: {json.dumps({'type': msg_type, 'content': content}, ensure_ascii=False)}\n\n"

    yield make_event("thought", "🤖 Agent 启动深度思考引擎...")

    # 💥 修改点 1：极其严苛的 System Prompt (紧箍咒)
    # 🌟 极其严苛的 System Prompt (紧箍咒)
    messages = [
        {"role": "system", "content": """你是一位严谨的动力电池诊断系统。
        【最高指令】：你必须、绝对、只能使用《知识库上下文》中提供的信息来回答问题！
        1. 仔细对比用户的提问与检索到的上下文内容。
        2. 只能提取和整合上下文中明确提到的实体、故障、症状和维修建议进行回答。
        3. 严禁进行任何发散、推理，绝不允许补充你自己的先验知识！
        4. 你的回答必须像是在做“语文阅读理解”，所有的结论、原因和建议，都必须能在上下文中找到对应原意。"""},
        {"role": "user", "content": user_question}
    ]

    MAX_TURNS = 5
    for turn in range(MAX_TURNS):
        response = await client_llm.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools_definition,
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        # 场景 A：得出最终结论
        if not response_message.tool_calls:
            answer = response_message.content
            print("==================================================")
            print(f"👩‍🔧 专家最终诊断:\n\n{answer}")
            print("==================================================")
            yield make_event("thought", "✅ 推理完成，正在生成诊断报告...")
            yield make_event("final", answer)
            return

        # 场景 B：大脑决定要调用工具
        messages.append(response_message)

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            try:
                function_args = json.loads(tool_call.function.arguments)
            except:
                function_args = {}

            if function_name == "tool_query_kg":
                semantic_query = function_args.get("semantic_query", "")
                yield make_event("thought", f"🔧 [工具] 启动混合检索引擎匹配语义: '{semantic_query}'...")
                function_response = await tool_query_kg(semantic_query)

            elif function_name == "tool_analyze_time_series":
                battery_id = function_args.get("battery_id", "")
                yield make_event("thought", f"📈 [工具] 跨服务呼叫 GPU 算力节点分析电池 {battery_id}...")

                raw_response = await tool_analyze_time_series(battery_id)

                # 💥 核心架构设计：数据分流 (Data Splitting)
                try:
                    res_dict = json.loads(raw_response)
                    if "plot_data" in res_dict:
                        # 1. 把海量数组抽出来，不让大模型看到，省钱且防幻觉
                        plot_data = res_dict.pop("plot_data")
                        # 2. 包装成一个全新的 "chart" 事件，直接发射给前端
                        yield make_event("chart", plot_data)
                        # 3. 干净的诊断结果喂给大模型做反思
                        function_response = json.dumps(res_dict, ensure_ascii=False)
                    else:
                        function_response = raw_response
                except:
                    function_response = raw_response

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })

        # 注入反思 Prompt

        reflection_prompt = {
            "role": "user",
            "content": f"系统提示：请严格基于上述工具返回的具体文字内容回答用户的问题。切勿脑补、推测或添加外部知识！如果返回的内容中包含了答案所需的症状、原因或建议，请直接提炼并输出完整回答。"
        }
        messages.append(reflection_prompt)
        print(f"🧐 [反思节点触发] Agent 正在对工具返回的数据进行交叉验证与物理常识审查...")
        yield make_event("thought", f"🧐 [反思] 第 {turn + 1} 轮数据获取完毕，正在进行物理常识审查与交叉验证...")
        print(f"🔄 [第 {turn + 1} 轮分析完毕] Agent 正在整合新数据，进行下一步推理...")

    fallback_answer = "抱歉，该问题过于复杂，系统经过多次诊断仍未得出确定结论，请转交人工专家处理。"
    print("==================================================")
    print(f"⚠️ 触发熔断:\n{fallback_answer}")
    print("==================================================")
    yield make_event("final", fallback_answer)
    return


# ==========================================
# 🚀 本地独立测试入口
# ==========================================
async def main():
    try:
        # 💥 修复本地直接运行 graph_rag_agent.py 报错的问题
        # 专门写一个壳子去消耗生成器的数据，这样就能在终端看到你的 print 特效了
        async def run_local_test(query):
            async for _ in agent_chat_stream(query):
                pass  # 数据已经在上面的 print 里打在终端了，这里只负责消耗 stream

        # 究极测试：先发现异常，再去查图谱，并且带反思
        await run_local_test("我的电池 B008 报了异常，它内部好像长了点毛刺，你能帮我分析一下原因吗？")

    finally:
        await kg_client.close()


if __name__ == "__main__":
    asyncio.run(main())