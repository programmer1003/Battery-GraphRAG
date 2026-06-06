# 使用官方极其轻量级的 Python 3.10 镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 把当前文件夹下的所有代码复制到容器里
COPY . /app

# 安装所有的依赖包 (建议你在根目录建一个 requirements.txt 包含 openai, neo4j, fastapi 等)
RUN pip install --no-cache-dir -r requirements.txt

# 暴露 FastAPI 的 8000 端口和 Streamlit 的 8501 端口
EXPOSE 8000 8501

# 默认启动命令（可以通过 docker-compose 覆盖）
CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]