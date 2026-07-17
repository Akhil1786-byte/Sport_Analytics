import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/player_stats.json")


def init_db():
    
    DB_PATH.parent.mkdir(exist_ok=True)
    if not DB_PATH.exists():
        DB_PATH.write_text(json.dumps({"players": {}}, indent=2))


def _load_data() -> dict:
    if not DB_PATH.exists():
        return {"players": {}}
    with open(DB_PATH, "r") as f:
        return json.load(f)


def _save_data(data: dict):
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_or_create_player(name: str) -> str:
    
    data = _load_data()
    if name not in data["players"]:
        data["players"][name] = []
        _save_data(data)
    return name


def save_match_stats(player_name: str, video_filename: str, stats: dict, segments: list = None):
    
    data = _load_data()

    if player_name not in data["players"]:
        data["players"][player_name] = []

    record = {
        "video_filename": video_filename,
        "match_date": datetime.now().isoformat(),
        "distance_covered": stats.get("distance_covered"),
        "avg_speed": stats.get("avg_speed"),
        "max_speed": stats.get("max_speed"),
        "avg_shot_speed": stats.get("avg_shot_speed"),
        "max_shot_speed": stats.get("max_shot_speed"),
        "consistency_score": stats.get("consistency_score"),
        "shot_count": stats.get("shot_count"),
        "segments": segments or [],
    }

    data["players"][player_name].append(record)
    _save_data(data)


def get_player_history(player_name: str) -> list[dict]:
    
    data = _load_data()
    history = data["players"].get(player_name, [])
    
    return sorted(history, key=lambda r: r["match_date"])
