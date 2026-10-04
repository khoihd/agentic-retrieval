import numpy as np
import ollama

MODEL = "qwen3:4b"
# MODEL = "qwen3:8b"

SYSTEM_PROMPT = """
You are an assistant with access to tools.
Use search_documents when the user's question depends on internal company
information, policies, procedures, rules, limits, or requirements.

Use search_memory when the question depends on information remembered from
previous interactions, such as user preferences, previous decisions, or
project-specific facts.

Do not guess company-specific information from general knowledge.
When a question has multiple information needs, make sure you have enough
information to answer each part. If a tool result answers only part of the
question, search again with a reformulated query for the missing information.
"""

def embed_text(text: str) -> list[float]:
    response = ollama.embed(model="nomic-embed-text", input=text)

    return response["embeddings"]


def cosine_similarity(a, b):
    a = np.squeeze(a)
    b = np.squeeze(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
