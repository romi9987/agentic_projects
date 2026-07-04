import asyncio
from openai import AsyncOpenAI
from rich import print
from rich.panel import Panel
from rich.console import Console

from react_agent import run_react_agent

console = Console()

async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="not-needed"
    )

    print("Welcome to the Travel React Agent!")
    print("Tell me what trip you want to plan.\n")

    # 1. Get user request interactively
    user_request = input("Your trip request: ")

    console.print("\n[bold green]Starting agent...[/bold green]\n")

    # 2. Run the agent with the user's request
    state = await run_react_agent(client, user_request)

    print(Panel("=== AGENT THOUGHTS ===", style="bold blue"))
    for t in state.thoughts:
        print("-", t)

    print(Panel("=== OBSERVATIONS ===", style="bold blue"))
    for o in state.observations:
        print("-", o)

    print("\n=== ERRORS ===")
    for e in state.errors:
        print("-", e)
    
    console.print(f"\n[bold green]Agent completed after {len(state.thoughts)} steps.[/bold green]")

if __name__ == "__main__":
    asyncio.run(main())
