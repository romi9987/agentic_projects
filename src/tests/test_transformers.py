from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

result = generator(
    "Agentic AI is a future, because",
    max_length=30,
    num_return_sequences=1,
)

print("Wynik:", result[0]["generated_text"])
