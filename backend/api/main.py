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

app = FastAPI(title="PokieTicker", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:7777", "http://127.0.0.1:7777"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
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
    ensure_market_index_table()
    ensure_ohlc_a_share_columns()
    ensure_layer1_event_columns()
    ensure_ticker_alias_table()
    ensure_news_aligned_attribution_columns()


@app.get("/api/health")
def health():
    return {"status": "ok"}
