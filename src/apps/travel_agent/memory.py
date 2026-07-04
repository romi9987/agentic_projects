import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MEMORY_DIR = SCRIPT_DIR / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_PATH = MEMORY_DIR / "memory.jsonl"

def write_memory(entry: dict):
    with MEMORY_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def load_memory() -> list[dict]:
    if not MEMORY_PATH.exists():
        return []
    with MEMORY_PATH.open() as f:
        return [json.loads(line) for line in f]
