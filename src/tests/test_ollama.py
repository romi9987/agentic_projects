import subprocess

prompt = "Explain agentic AI in simple words."

result = subprocess.run(
    ["ollama", "run", "llama3"], input=prompt.encode("utf-8"), stdout=subprocess.PIPE
)

print("Ollama odpowiedziała:\n", result.stdout.decode("utf-8"))
