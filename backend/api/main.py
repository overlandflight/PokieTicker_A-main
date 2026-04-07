from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.database import (
    check_db_connection,
    ensure_layer1_event_columns,
    ensure_market_index_table,
    ensure_news_aligned_attribution_columns,
    ensure_ohlc_a_share_columns,
    ensure_ticker_alias_table,
)
from backend.api.routers import stocks, news, analysis, predict, pipeline
from backend.init_db import init_database   # 新增导入

app = FastAPI(title="PokieTicker", version="1.0.0")

# 允许所有来源的跨域请求（生产环境可限制为具体前端域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(news.router, prefix="/api/news", tags=["news"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(predict.router, prefix="/api/predict", tags=["predict"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])


@app.on_event("startup")
def startup():
    errors = settings.validate_for_startup()
    if errors:
        raise RuntimeError("Startup configuration invalid:\n- " + "\n- ".join(errors))
    check_db_connection()
    
    # 🔥 新增：自动执行 init.sql 建表（解决 Railway 部署时表缺失问题）
    init_database()
    
    # 原有的列确保函数（不会与建表冲突，可保留）
    ensure_market_index_table()
    ensure_ohlc_a_share_columns()
    ensure_layer1_event_columns()
    ensure_ticker_alias_table()
    ensure_news_aligned_attribution_columns()


@app.get("/api/health")
def health():
    return {"status": "ok"}
