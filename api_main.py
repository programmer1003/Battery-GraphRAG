from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

# 导入你已经写好的核心逻辑
from graph_rag_agent import graph_rag_chat, get_dynamic_fault_list

app = FastAPI(
    title="🔋 动力电池 GraphRAG 智能诊断引擎",
    description="提供基于 Neo4j 云原生图谱的零幻觉多跳诊断 API",
    version="1.0.0"
)

# 定义前端传过来的数据格式
class DiagnoseRequest(BaseModel):
    question: str

# 接口 1：探活与感知接口
@app.get("/health")
async def health_check():
    """获取当前云端图谱已知的故障列表，证明系统存活"""
    known_faults = get_dynamic_fault_list()
    return {"status": "ok", "known_faults_count": len(known_faults), "faults": known_faults}

# 接口 2：核心诊断接口
@app.post("/api/diagnose")
async def diagnose_fault(request: DiagnoseRequest):
    """接收用户提问，执行图谱多跳检索并返回专家建议"""
    print(f"\n[API 收到请求] 问题: {request.question}")
    try:
        # 直接调用你原来的核心函数
        answer = graph_rag_chat(request.question)
        return {
            "status": "success",
            "question": request.question,
            "answer": answer
        }
    except Exception as e:
        print(f"[API 报错] {e}")
        raise HTTPException(status_code=500, detail="诊断引擎内部错误，请检查日志。")

if __name__ == "__main__":
    print("🚀 正在启动 FastAPI 后端微服务 (端口: 8000)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)