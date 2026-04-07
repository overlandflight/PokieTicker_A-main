import pymysql
from pathlib import Path
from backend.config import settings

def init_database():
    """
    在应用启动时读取项目根目录的 init.sql 并执行建表语句。
    如果表已存在，会自动跳过（CREATE TABLE IF NOT EXISTS）。
    """
    # 获取数据库连接参数
    conn = pymysql.connect(
        host=settings.MYSQL_HOST,
        user=settings.MYSQL_USER,
        password=settings.MYSQL_PASSWORD,
        database=settings.MYSQL_DATABASE,
        charset='utf8mb4'
    )
    cursor = conn.cursor()

    # 定位 init.sql 文件（项目根目录）
    # __file__ = backend/init_db.py → 向上两级到项目根目录
    sql_path = Path(__file__).parent.parent / "init.sql"
    if not sql_path.exists():
        print(f"⚠️ 未找到 init.sql 文件，跳过自动建表。预期路径: {sql_path}")
        return

    with open(sql_path, 'r', encoding='utf-8') as f:
        sql_content = f.read()

    # 按分号分割并逐条执行
    for statement in sql_content.split(';'):
        stmt = statement.strip()
        if stmt and not stmt.startswith('--'):
            try:
                cursor.execute(stmt)
            except Exception as e:
                # 忽略表已存在等重复创建错误
                print(f"执行SQL片段时出现警告（可忽略）: {e}")

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ 数据库表初始化完成（如果表不存在则已创建）")
