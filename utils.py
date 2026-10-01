import numpy as np
import os
from dotenv import load_dotenv
from google import genai
from smolagents import LiteLLMModel

client = None

def embed_text(text):
    client = get_client()
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )

    return response.embeddings[0].values

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_client() -> genai.Client:
    global client
    if client:
        return client
    
    load_dotenv()
    api_key = os.environ["GEMINI_API_KEY"]
    print("API key loaded:", bool(api_key))
    client = genai.Client(api_key=api_key)
    return client


def get_local_agent():
    model = LiteLLMModel(
        model_id="ollama_chat/qwen3:4b",  # Or try other Ollama-supported models
        api_base="http://127.0.0.1:11434",  # Default Ollama local server
        num_ctx=8192,
    )

    return model