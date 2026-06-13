import json
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Optional


class MemoryStore:
    """
    Persists conversation turns to a JSON file across sessions.
    Acts as long-term memory for the agent.
    """
# Key design decisions:
    # 1. JSON file format: Simple, human-readable, and Colab-friendly. 
    #   For production, you’d use a proper database or vector store.
    # 2. Append-only writes: Each conversation turn is appended, creating a complete audit trail.
    # 3. Lazy loading: We only load from disk when needed, keeping memory footprint low.
    # 4. Graceful degradation: If the file is corrupted or missing, we return an empty list rather than crashing.

    def __init__(self, file_path: str, max_entries: int = 50):
        self.file_path = file_path
        self.max_entries = max_entries
        self._ensure_file()
    
    def _ensure_file(self):
        path = Path(self.file_path)
        path.parent.mkdir(parents=True, exist_ok=True)  # ← creates src/apps/memory/ if needed
        if not path.exists():
            with open(path, "w") as f:
                json.dump([], f)
    
    def load_all(self) -> List[dict]:
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except Exception:
            return []
    
    def append(self, entry: dict):
        data = self.load_all()
        data.append(entry)
        # Trim to max_entries so the file never grows beyond max_entries
        if len(data) > self.max_entries:
            data = data[-self.max_entries:]        
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_recent(self, limit: Optional[int] = None) -> list[dict]:
        data = self.load_all()
        limit = limit or self.max_entries
        return data[-limit:]
    
    def delete_all(self):
        with open(self.file_path, "w") as f:
            json.dump([], f)

    def save_turn(self, session_id: str, user_input: str, agent_answer: str):
        """Convenience method — saves both sides of a completed turn at once."""
        timestamp = datetime.now(UTC).isoformat()
        self.append({
            "session_id": session_id,
            "timestamp": timestamp,
            "role": "user",
            "content": user_input,
        })
        self.append({
            "session_id": session_id,
            "timestamp": timestamp,
            "role": "assistant",
            "content": agent_answer,
        })
