from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

@dataclass(frozen=True)
class Settings:
    client_id: str
    client_secret: str
    base_url: str = "https://openapi.tossinvest.com"
    universe_path: Path = ROOT / "data" / "universe.csv"
    history_dir: Path = ROOT / "data" / "history"
    results_dir: Path = ROOT / "data" / "results"

def get_settings() -> Settings:
    cid = os.getenv("TOSS_CLIENT_ID", "")
    secret = os.getenv("TOSS_CLIENT_SECRET", "")
    if not cid or not secret:
        raise RuntimeError(
            "TOSS_CLIENT_ID / TOSS_CLIENT_SECRET가 없습니다. .env 파일을 확인하세요."
        )
    return Settings(
        client_id=cid,
        client_secret=secret,
        base_url=os.getenv("TOSS_BASE_URL", "https://openapi.tossinvest.com"),
    )
