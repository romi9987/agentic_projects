import time
import redis
import json
import streamlit as st
from datetime import datetime

# Connect to Redis (adjust host/port/password if remote)
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

MAX_ARCHIVES = 1  # keep only the last N archives per user


def save_state(session_id: str, state: dict):
    key = f"chat_state:{session_id}"
    data = {
        "summary": state["summary"],
        "display_history": state["display_history"],
    }
    r.set(key, json.dumps(data))


def load_state(session_id: str) -> dict:
    key = f"chat_state:{session_id}"
    data = r.get(key)
    if data:
        parsed = json.loads(data)
        return {
            "summary": parsed.get("summary", ""),
            "history": [],  # short-term context stays fresh each run
            "display_history": parsed.get("display_history", []),
        }
    return {"summary": "", "history": [], "display_history": []}


def archive_chat(session_id: str, state: dict):
    # don't archive empty chats
    if not state.get("display_history"):
        return

    # delete previous archive if user is continuing from one
    old_keys = sorted(r.keys(f"chat_archive:{session_id}:*"), reverse=True)
    if old_keys:
        # remove the most recent one (we'll replace it)
        r.delete(old_keys[0])

    # save new archive with fresh timestamp
    ts = int(time.time())
    key = f"chat_archive:{session_id}:{ts}"
    r.set(key, json.dumps(state))

    # cleanup: keep only last MAX_ARCHIVES
    keys = sorted(r.keys(f"chat_archive:{session_id}:*"), reverse=True)
    if len(keys) > MAX_ARCHIVES:
        for k in keys[MAX_ARCHIVES:]:
            r.delete(k)

    # clear active state
    r.delete(f"chat_state:{session_id}")


def show_archive(session_id: str):
    keys = r.keys(f"chat_archive:{session_id}:*")
    for k in sorted(keys, reverse=True):
        data = json.loads(r.get(k))
        ts = int(k.split(":")[-1])
        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")

        # first user message as preview
        preview = ""
        for msg in data.get("display_history", []):
            if msg["role"] == "user":
                preview = msg["content"][:40] + "..."
                break
        label = f"{dt} – {preview}" if preview else f"{dt}"

        if st.button(f"▶️ Load {label}", key=f"load-{k}"):
            st.session_state.graph_state = {
                "summary": data.get("summary", ""),
                "history": [],  # rebuild rolling context naturally
                "display_history": data.get("display_history", []),
            }
            st.success(f"Loaded archived chat from {dt} ✅")
            st.rerun()


def delete_all_archives(session_id: str):
    # delete all archives for this session
    keys = r.keys(f"chat_archive:{session_id}:*")
    for k in keys:
        r.delete(k)
    # also delete active chat
    r.delete(f"chat_state:{session_id}")
