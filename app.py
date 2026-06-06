import streamlit as st
import os
from pdf_to_graph import extract_text_from_pdf, llm_extract_triplets, inject_triplets_to_neo4j
from graph_rag_agent import graph_rag_chat

st.set_page_config(page_title="电池维修专家系统", layout="wide")

st.title("🔋 动力电池智能故障诊断专家系统")

# 侧边栏：知识注入区
with st.sidebar:
    st.header("📚 知识库更新 (PDF上传)")
    uploaded_file = st.file_uploader("上传维修手册 (PDF)", type="pdf")
    if uploaded_file and st.button("开始自动化学习"):
        with st.spinner("Agent 正在深度阅读并更新图谱..."):
            # 保存临时文件
            temp_path = "temp.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 自动化管线
            text = extract_text_from_pdf(temp_path)
            triplets = llm_extract_triplets(text)
            inject_triplets_to_neo4j(triplets)
            st.success(f"成功注入 {len(triplets)} 条知识！")
            os.remove(temp_path)

# 主界面：对话区
st.header("🤖 智能排障对话")
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理用户输入
if prompt := st.chat_input("请描述您遇到的电池故障现象..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在查询云端图谱并生成诊断建议..."):
            # 调用你之前写的推理核心
            response = graph_rag_chat(prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})