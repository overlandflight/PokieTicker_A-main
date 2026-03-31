#!/bin/bash

# 等待 MySQL 就绪（如果使用了 Railway MySQL 插件）
if [ -n "$MYSQL_HOST" ]; then
    echo "Waiting for MySQL..."
    # 使用 mysql 客户端检查连接（需安装 mysql-client，但镜像中无，改用 nc 或直接等待）
    # 这里简单等待 5 秒，因为 Railway 的 MySQL 通常启动较快
    sleep 5
    # 可选：执行数据库初始化
    if [ -f "init.sql" ]; then
        echo "Initializing database..."
        # 使用 mysql 命令行客户端（需安装 default-mysql-client）
        # 如果未安装，可以注释掉，手动在 Railway 控制台执行 init.sql
        # 这里假设 Railway 的 Python 镜像中没有 mysql 客户端，所以跳过自动初始化
        # 改为提示用户手动执行
        echo "Please manually run init.sql in Railway MySQL console."
    fi
fi

# 启动应用
exec uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}