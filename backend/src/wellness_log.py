import json
import os
from datetime import datetime

FILE = "wellness_log.json"


def load_log():
    if not os.path.exists(FILE):
        return []
    with open(FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_last_entry():
    log = load_log()
    if not log:
        return None
    return log[-1]


def add_entry(mood, energy, goals, summary):
    log = load_log()
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "mood": mood,
        "energy": energy,
        "goals": goals,
        "summary": summary
    }
    log.append(entry)
    with open(FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
