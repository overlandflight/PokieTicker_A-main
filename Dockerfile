FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖（AkShare 可能需要的编译工具）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装 Python 包
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有后端代码
COPY . .

# 复制启动脚本并设置权限
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 暴露端口（Railway 会自动用 $PORT 覆盖）
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]