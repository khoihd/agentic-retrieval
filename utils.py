import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import ollama

# MODEL = "qwen3:4b"
# MODEL = "qwen3:8b"
MODEL = "qwen3.5:4b"
# MODEL = "qwen3.5:9b"

MAX_MEMORY_RETRIEVAL = 20

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

load_dotenv()
client = None

def get_client():
    global client
    if client:
        return client 
    
    # Local Ollama
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    return client

def embed_text(text: str) -> list[float]:
    response = ollama.embed(model="nomic-embed-text", input=text)

    return response["embeddings"][0]


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

