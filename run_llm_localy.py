# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="unsloth/Qwen3-0.6B-unsloth-bnb-4bit")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)