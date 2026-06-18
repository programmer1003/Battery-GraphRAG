FROM python:3.10-slim

WORKDIR /app

# 优先拷贝 requirements.txt，利用 Docker 缓存层加速构建
COPY requirements.txt /app/

# 💡 极其硬核的优化：强制安装纯 CPU 版本的 PyTorch，将镜像体积缩小几个 G！
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 安装其他所有依赖
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝所有代码文件和数据集、模型权重
COPY . /app/

# 暴露三个微服务端口
EXPOSE 8000 8001 8501