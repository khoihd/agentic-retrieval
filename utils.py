import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
EMBEDDING_MODEL = "mlx-community/bge-small-en-v1.5-4bit"

MAX_MEMORY_RETRIEVAL = 20
DEFAULT_MEMORY_SEARCH_TOP_K = 3
MEMORY_RELATIONSHIP_TOP_K = 10


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

    # Local MLX
    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="mlx",
    )
    
    print(f"Running model {MODEL}")
    print(f"Running embedding model {EMBEDDING_MODEL}")

    return client

def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)

    return response.data[0].embedding


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

