import pymysql
from backend.config import get_db_config

def init_database():
    cfg = get_db_config()
    conn = pymysql.connect(
        host=cfg['host'],
        user=cfg['user'],
        password=cfg['password'],
        database=cfg['database'],
        charset='utf8mb4'
    )
    cursor = conn.cursor()
    
    # 读取 init.sql
    with open('init.sql', 'r', encoding='utf-8') as f:
        sql_script = f.read()
    
    # 按分号分割执行（注意处理注释和空语句）
    for statement in sql_script.split(';'):
        if statement.strip() and not statement.strip().startswith('--'):
            try:
                cursor.execute(statement)
            except Exception as e:
                print(f"忽略已存在表错误: {e}")
    conn.commit()
    cursor.close()
    conn.close()
