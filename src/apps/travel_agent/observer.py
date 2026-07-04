import json
from datetime import datetime
from pathlib import Path

# Choose your log directory - relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Create a timestamped log file
LOG_PATH = LOG_DIR / f"react_agent_{datetime.now().isoformat()}.jsonl"

# Open the file
_log_file = LOG_PATH.open("a", encoding="utf-8")

def write_jsonl(event: dict):
    if '_log_file' in globals() and _log_file:
        _log_file.write(json.dumps(event, default=str) + "\n")
        _log_file.flush()

def instrument_node(name: str):
    """
    Decorator to instrument async pipeline nodes.
    Writes JSONL logs only.
    """
    def decorator(fn):
        async def wrapper(*args, **kwargs):
            write_jsonl({"event": "node.start", "node": name})

            try:
                result = await fn(*args, **kwargs)
                write_jsonl({
                    "event": "node.success",
                    "node": name,
                    "result": result,
                })
                return result

            except Exception as e:
                write_jsonl({
                    "event": "node.error",
                    "node": name,
                    "error": str(e),
                })
                raise

        return wrapper
    return decorator
