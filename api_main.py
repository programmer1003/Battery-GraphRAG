# api_main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse # 💥 必须引入这个
from pydantic import BaseModel
import uvicorn

# 💥 引入带 yield 的流式大脑函数
from graph_rag_agent import agent_chat_stream

app = FastAPI(title="🔋 动力电池诊断引擎", version="2.0.0")

class DiagnoseRequest(BaseModel):
    question: str

@app.post("/api/diagnose")
async def diagnose_fault(request: DiagnoseRequest):
    print(f"\n[API 收到请求] 问题: {request.question}")
    try:
        # 💥 核心修复：直接把函数塞进去，千万不要加 await！
        # 因为 StreamingResponse 内部会自己去一边等一边抽水（消费生成器）
        return StreamingResponse(
            agent_chat_stream(request.question),
            media_type="text/event-stream"
        )
    except Exception as e:
        print(f"[API 报错] {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)