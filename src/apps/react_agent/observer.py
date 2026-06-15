import json
import os
import uuid
import time
from pathlib import Path

class AgentObserver:
    # AgentObserver is a tracing tool — it records what happened and how long it took, 
    # structured as events you can analyse programmatically. 
    # Each run gets its own JSONL file you can parse, query, or feed into a dashboard.

    # What AgentObserver does well:
        # Per-run trace files — easy to compare run A vs run B
        # Span timing — you can see exactly how long each LLM call or tool execution took
        # Structured JSONL — queryable with pandas, jq, or any JSON tool
        # Completely self-contained — no external dependency

    # What it lacks vs loguru:
        # No log levels (no DEBUG/INFO/WARNING/ERROR distinction)
        # No console output — blind while developing unless you add print statements
        # No log rotation — trace files accumulate forever
        # No colour/formatting for human reading
    def __init__(self, log_dir=os.getenv("AGENT_OBSERVER_DIR")):
        # We assign a unique trace ID (UUID) to every run. 
        # This makes it easy to correlate logs and see exactly what happened during a specific session.
        self.trace_id = str(uuid.uuid4())
        self.events = []
        
        Path(log_dir).mkdir(exist_ok=True)
        # We use JSONL, where each line is a complete JSON object. 
        # This format keeps parsing simple, even for massive logs.
        self.file_path = Path(log_dir) / f"trace_{self.trace_id}.jsonl"
    
    def log(self, event_type, data=None):
        # Every log entry follows a structured schema: 
        # it always includes a type, a timestamp, and any additional arbitrary data. 
        # This makes analysis and debugging straightforward.
        entry = {
            "trace_id": self.trace_id,
            "timestamp": time.time(),
            "event": event_type,
            "data": data or {}
        }
        
        self.events.append(entry)
        
        with open(self.file_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def span(self, name):
        return Span(self, name)
    

class Span:
    """Spans are critical for understanding timing and performance"""
    def __init__(self, observer, name):
        self.observer = observer
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        self.observer.log("span_start", {"name": self.name})
    
    def __exit__(self, exc_type, exc, tb):
        duration = time.time() - self.start
        self.observer.log("span_end", {
            "name": self.name,
            "duration_sec": round(duration, 3)
        })
