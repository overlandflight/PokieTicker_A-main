from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yml"


def _load_yaml() -> tuple[dict[str, Any], list[str]]:
    if not CONFIG_PATH.exists():
        return {}, [f"Config file not found: {CONFIG_PATH}"]

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        return {}, [f"Config file is invalid YAML: {exc}"]

    if data is None:
        return {}, []
    if not isinstance(data, dict):
        return {}, [f"Config file must contain a YAML object at the root: {CONFIG_PATH}"]
    return data, []


def _section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key, {})
    return value if isinstance(value, dict) else {}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class Settings:
    config_path: Path
    config_errors: list[str] = field(default_factory=list)

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-reasoner"
    polygon_api_key: str = ""

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "pokieticker"
    mysql_charset: str = "utf8mb4"

    @classmethod
    def from_config(cls, cfg: dict[str, Any], errors: list[str]) -> "Settings":
        # 优先从环境变量读取
        # MySQL
        mysql_host = os.environ.get("MYSQL_HOST") or os.environ.get("DB_HOST")
        mysql_port = os.environ.get("MYSQL_PORT") or os.environ.get("DB_PORT")
        mysql_user = os.environ.get("MYSQL_USER") or os.environ.get("DB_USER")
        mysql_password = os.environ.get("MYSQL_PASSWORD") or os.environ.get("DB_PASSWORD")
        mysql_database = os.environ.get("MYSQL_DATABASE") or os.environ.get("DB_NAME") or os.environ.get("RAILWAY_MYSQL_DATABASE")
        # DeepSeek
        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        deepseek_base_url = os.environ.get("DEEPSEEK_BASE_URL")
        deepseek_model = os.environ.get("DEEPSEEK_MODEL")
        # Polygon
        polygon_api_key = os.environ.get("POLYGON_API_KEY")

        # 如果环境变量未提供，再使用 YAML 配置
        if not mysql_host:
            mysql_cfg = _section(cfg, "mysql")
            mysql_host = mysql_cfg.get("host", "127.0.0.1")
            mysql_port = mysql_cfg.get("port", 3306)
            mysql_user = mysql_cfg.get("user", "root")
            mysql_password = mysql_cfg.get("password", "")
            mysql_database = mysql_cfg.get("database", "pokieticker")

        if not deepseek_api_key:
            deepseek_cfg = _section(cfg, "deepseek")
            deepseek_api_key = deepseek_cfg.get("api_key", "")
            deepseek_base_url = deepseek_cfg.get("base_url", "https://api.deepseek.com")
            deepseek_model = deepseek_cfg.get("model", "deepseek-reasoner")

        if not polygon_api_key:
            polygon_cfg = _section(cfg, "polygon")
            polygon_api_key = polygon_cfg.get("api_key", "")

        raw_base_url = str(deepseek_base_url or "https://api.deepseek.com")
        return cls(
            config_path=CONFIG_PATH,
            config_errors=list(errors),
            deepseek_api_key=str(deepseek_api_key or ""),
            deepseek_base_url=raw_base_url.replace("/chat/completions", "").rstrip("/"),
            deepseek_model=str(deepseek_model or "deepseek-reasoner"),
            polygon_api_key=str(polygon_api_key or ""),
            mysql_host=str(mysql_host or "127.0.0.1"),
            mysql_port=_as_int(mysql_port, 3306),
            mysql_user=str(mysql_user or "root"),
            mysql_password=str(mysql_password or ""),
            mysql_database=str(mysql_database or "pokieticker"),
            mysql_charset=str(mysql_cfg.get("charset", "utf8mb4") if 'mysql_cfg' in locals() else "utf8mb4"),
        )

    def validate_for_startup(self) -> list[str]:
        errors = list(self.config_errors)
        if not self.mysql_host.strip():
            errors.append("mysql.host must not be empty")
        if not self.mysql_user.strip():
            errors.append("mysql.user must not be empty")
        if not self.mysql_database.strip():
            errors.append("mysql.database must not be empty")
        return errors


_cfg, _cfg_errors = _load_yaml()
settings = Settings.from_config(_cfg, _cfg_errors)