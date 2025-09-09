Here I want to present my Agentic AI Projects:

1. Chatbot - llama3 based chatbot with follow-up conversation and chat archive in streamlit ui.

2. Info_Retrieval - llama3 based 2-agent system retrieving information from documents in streamlit ui.

How to install:
uv venv
source .venv/bin/activate
uv sync # installs both main and dev dependencies
uv sync --no-dev # installs only dev dependencies

ollama pull llama3
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
