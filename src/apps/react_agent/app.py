import os
from dotenv import load_dotenv
from openai import OpenAI

from tools import registry, Tool, make_delete_all_memory_tool, DeleteAllMemoryArgs
from llm_wrapper import ReactAgent
from logger import log, setup_logger
from memory import MemoryStore


load_dotenv()

setup_logger(
    log_file=os.getenv("AGENT_LOG_PATH"),
    verbose=True,
)

log.info("Agent starting up")

def run_once(agent: ReactAgent, task: str):
    """Run the agent on a single task and print the result."""
    print(f"\n>>> TASK:\n{task.strip()}\n")
    answer = agent.run(task)
    print(f"\n{'='*48}")
    print("  FINAL ANSWER")
    print(f"{'='*48}\n")
    print(answer)


def chat_loop(agent: ReactAgent):
    """Interactive chat loop."""
    print("\nReAct Agent — type 'exit' to quit.\n")
    while True:
        task = input("You: ").strip()
        if task.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if not task:
            continue
        try:
            answer = agent.run(task)
            print(f"\nAgent: {answer}\n")
        except RuntimeError as e:
            print(f"[Agent error] {e}")
        except Exception as e:
            print(f"[Unexpected error] {e}")


if __name__ == "__main__":

    client = OpenAI(
        # base_url="http://localhost:8000/v1", # base_url for omlx
        base_url="http://localhost:1234/v1", # base_url for lmstudio
        api_key="omlx",  # required by the library, ignored by Omlx
    )

    # We create a global memory store that persists to disk
    memory_store = MemoryStore(
        file_path=os.getenv("AGENT_MEMORY_PATH", "agent_memory.json"),  # fallback to local
        # The max_entries parameter controls how many recent conversations we inject into context. 
        # Setting this to 50 means we'll load the last 50 user-assistant exchanges, 
        # which is typically 2500-10000 tokens.
        max_entries=50,
    )

    # Register make_delete_all_memory_tool here, not in tools.py.
    # This pattern — defining the tool factory in tools.py but registering it in app.py — is the right approach 
    # for any tool that depends on external state like memory_store, a database connection, or an API client. 
    # It keeps tools.py free of dependencies it shouldn't know about.
    registry.register(Tool(
        name="delete_all_memory",
        description="Permanently deletes all long-term memory. " \
                    "This action is irreversible and requires human approval before execution.",
        input_schema=DeleteAllMemoryArgs,
        func=make_delete_all_memory_tool(memory_store),
            ),
        destructive=True,   # ← flag destructive tool here
        )

    agent = ReactAgent(
        client=client,
        registry=registry,
        model="qwen_qwen3-coder-next",
        max_iterations=10,
        verbose=True,
        memory_store=memory_store,       # pass memory in
        memory_injection_limit=10,       # inject last 10 turns as context
    )
    
    # --- Single task demo ---
    task = """
    What's today's date?
    What is 15% of 847?
    What's the weather in Gdansk?
    What are the latest developments in Claude AI models?
    """
    # run_once(agent, task)

    # --- Drop into interactive chat after the demo ---
    chat_loop(agent)
