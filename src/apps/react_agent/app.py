from openai import OpenAI
from tools import registry
from llm_wrapper import ReactAgent


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
        base_url="http://localhost:8000/v1",
        api_key="omlx",  # required by the library, ignored by Omlx
    )

    agent = ReactAgent(
        client=client,
        registry=registry,
        model="Qwen3.6-35B-A3B-4bit",
        max_iterations=10,
        verbose=True,
    )

    # --- Single task demo ---
    task = """
    What's today's date?
    What is 15% of 847?
    What's the weather in Gdansk?
    What are the latest developments in Claude AI models?
    """
    run_once(agent, task)

    # --- Drop into interactive chat after the demo ---
    chat_loop(agent)
