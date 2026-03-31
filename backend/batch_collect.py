"""Check and display batch processing status.

Usage: python -m backend.batch_collect [batch_id]

注意: DeepSeek 不支持 Anthropic Batch API，
此模块简化为查看批处理任务状态。
"""

import sys

from backend.database import get_conn


def main():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM batch_jobs ORDER BY created_at DESC")
            jobs = cur.fetchall()
    finally:
        conn.close()

    if not jobs:
        print("No batch jobs found.")
        print("\nNote: DeepSeek API processes synchronously via 'python -m backend.batch_submit'")
        return

    print("Batch jobs:")
    for j in jobs:
        print(f"  {j['batch_id']}  status={j['status']}  total={j['total']}  "
              f"completed={j['completed']}  created={j['created_at']}")


if __name__ == "__main__":
    main()
