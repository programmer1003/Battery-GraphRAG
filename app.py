# app.py (纯前端网络交互版 - 支持 SSE 流式打字机特效与历史状态保持)
import streamlit as st
import requests
import json
import pandas as pd

st.set_page_config(page_title="电池维修专家系统", layout="wide")
st.title("🔋 动力电池智能故障诊断专家系统")

st.sidebar.info(
    "🌐 当前系统架构：前端界面 (Port 8501) -> Agent 网关 (Port 8000) -> TF-GDC 算法引擎 (Port 8001) + Neo4j 混合图谱"
)

# 主界面：对话区
st.header("🤖 智能排障对话")
if "messages" not in st.session_state:
    st.session_state.messages = []

# 💥 优化点 1：完美还原历史聊天记录（包括文本和物理波形图）
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 如果历史消息里存了图表数据，重新把它画出来
        if "charts" in message and message["charts"] is not None:
            chart_df = pd.DataFrame(message["charts"])
            with st.expander("📊 历史诊断时序数据切片", expanded=False):
                if "Voltage_V" in chart_df.columns:
                    st.caption("⚡ 单体电压降维特征 (Voltage_V)")
                    st.line_chart(chart_df["Voltage_V"], color="#FF4B4B", height=150)
                if "Temperature_C" in chart_df.columns:
                    st.caption("🌡️ 表面温度温升特征 (Temperature_C)")
                    st.line_chart(chart_df["Temperature_C"], color="#FF8C00", height=150)

# 处理用户输入
if prompt := st.chat_input("请描述您遇到的电池故障现象 (可指定电池编号如 B008)..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_container = st.status("🧠 Agent 正在深度思考中...", expanded=True)
        final_answer_placeholder = st.empty()
        answer = ""
        current_charts = None  # 💥 新增：用于暂存当前这轮对话抓取到的图表数据

        try:
            response = requests.post(
                "http://127.0.0.1:8000/api/diagnose",
                json={"question": prompt},
                stream=True,
                timeout=60
            )

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        try:
                            data = json.loads(decoded_line[6:])

                            if data["type"] == "thought":
                                status_container.write(data["content"])

                            elif data["type"] == "chart":
                                current_charts = data["content"]
                                chart_df = pd.DataFrame(current_charts)

                                status_container.markdown("📊 **TF-GDC 引擎截获的底层物理时序切片 (Window=512):**")

                                # 💥 优化点 2：增加列名校验，避免因 CSV 格式不一致导致前端红屏崩溃
                                if "Voltage_V" in chart_df.columns:
                                    status_container.caption("⚡ 单体电压降维特征 (Voltage_V)")
                                    status_container.line_chart(chart_df["Voltage_V"], color="#FF4B4B", height=150)
                                if "Temperature_C" in chart_df.columns:
                                    status_container.caption("🌡️ 表面温度温升特征 (Temperature_C)")
                                    status_container.line_chart(chart_df["Temperature_C"], color="#FF8C00", height=150)

                            elif data["type"] == "final":
                                status_container.update(label="✅ 诊断推理完成", state="complete", expanded=False)
                                answer = data["content"]
                                final_answer_placeholder.markdown(answer)

                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            status_container.update(label="❌ 网络连接异常", state="error")
            st.error(f"无法连接到 Agent 中枢 (Port 8000)，报错信息: {e}")

    # 💥 优化点 3：把最终文本和截获到的图表数据一起存进记忆里
    if answer:
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "charts": current_charts
        })