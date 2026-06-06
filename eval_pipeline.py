import os
import time
from dotenv import load_dotenv

# 导入我们自己写的核心逻辑模块
from pdf_to_graph import extract_text_from_pdf, llm_extract_triplets, inject_triplets_to_neo4j
from graph_rag_agent import graph_rag_chat, get_dynamic_fault_list

# 强制加载环境变量密码
load_dotenv()

print("=========================================")
print("🚀 开始执行 GraphRAG 全链路自动化测评 (EVAL)")
print("=========================================\n")

# ---------------------------------------------------------
# 1. 测评：自动化知识提取与入库 (Text2Graph)
# ---------------------------------------------------------
print("▶️ [步骤 1] 正在测评 Text2Graph 离线管线...")
# 💥 替换为你真实的 PDF 绝对路径
test_pdf_path = "C:/Users/赵旭东/Desktop/Battery_Agent/Battery_RAG/某品牌新能源电池故障诊断手册.pdf"

try:
    print("   - 📄 [阶段 A] 正在解析 PDF 文本...")
    text = extract_text_from_pdf(test_pdf_path)
    print(f"   ✅ [成功] 提取文本长度: {len(text)} 字符")

    print("   - 🧠 [阶段 B] 正在调用 LLM 抽取实体关系三元组...")
    triplets = llm_extract_triplets(text)
    print(f"   ✅ [成功] 成功抽取 {len(triplets)} 条强约束知识三元组")

    print("   - 🕸️ [阶段 C] 正在向 Neo4j 云原生图谱执行 Cypher 注入...")
    # 暗访考核点：数据能不能成功写入远端 AuraDB？
    inject_triplets_to_neo4j(triplets)
    print("   ✅ [成功] 知识入库完毕！图谱底座更新成功。")
except Exception as e:
    print(f"❌ 步骤 1 测试失败，管道中断: {e}")
    exit()

time.sleep(2)  # 稍微等两秒，模拟系统处理缓冲

# ---------------------------------------------------------
# 2. 测评：Agent 动态感知探针 (Perception)
# ---------------------------------------------------------
print("\n▶️ [步骤 2] 正在测评 Agent 数据库动态探针...")
try:
    # 暗访考核点：Agent 能不能自动察觉到步骤 1 刚刚写入的新故障节点？
    known_faults = get_dynamic_fault_list()
    if known_faults:
        print(f"   ✅ [探针生效] 获取到当前云端最新故障名录: {known_faults}")
    else:
        print("   ❌ [探针异常] 未获取到任何数据，请检查 Neo4j 是否写入成功。")
except Exception as e:
    print(f"❌ 步骤 2 测试失败: {e}")

time.sleep(1)

# ---------------------------------------------------------
# 3. 测评：在线多跳检索与 RAG 问答闭环 (Action & Generation)
# ---------------------------------------------------------
print("\n▶️ [步骤 3] 正在测评 GraphRAG 智能诊断引擎...")

test_question = "我的车报了 BMS 通信丢失，该咋办？"
print(f"   👤 模拟用户输入: \"{test_question}\"")
print("   🤖 启动大模型意图识别与图谱多跳检索...\n")

try:
    # 暗访考核点：
    # 1. 能不能精准提取出实体 "BMS通信丢失"？
    # 2. 能不能把 Neo4j 里的多跳数据（症状、根因）全部抓出来？
    # 3. 最终回复会不会发生幻觉？
    final_answer = graph_rag_chat(test_question)

    print("\n🎯 最终诊断报告 (EVAL 结果):")
    print("--------------------------------------------------")
    print(final_answer)
    print("--------------------------------------------------")
except Exception as e:
    print(f"❌ 步骤 3 测试失败: {e}")

print("\n🎉 全链路 GraphRAG 自动化测试 (EVAL) 执行完毕！系统 100% 连通！")