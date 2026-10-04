from search_documents import rerank_chunks, format_document_retrieval_context
from memory import rerank_memories, format_memory_retrieval_context
from google.genai import types
from utils import get_client

def build_retrieval_context(original_query, retrieved_chunks, retrieved_memories):
    chunk_seen = set()
    memory_seen = set()
    unique_chunks = []
    unique_memories = []

    for chunk in retrieved_chunks:
        chunk_id = chunk["chunk_id"]

        if chunk_id in chunk_seen:
            continue

        chunk_seen.add(chunk_id)
        unique_chunks.append(chunk)

    for memory in retrieved_memories:
        memory_id = memory['id']
        if memory_id in memory_seen:
            continue

        memory_seen.add(memory_id)
        unique_memories.append(memory)

    context_chunks = rerank_chunks(
        original_query,
        unique_chunks,
    )

    context_memories = rerank_memories(
        original_query,
        unique_memories
    )

    return context_chunks, context_memories


def generate_grounded_answer(user_query, context_documents, context_memories,):
    evidence = format_document_retrieval_context(context_documents)
    memory = format_memory_retrieval_context(context_memories)
    prompt = f"""
Answer the user's question using only the DOCUMENT EVIDENCE and MEMORY provided below.

Rules:
- Use only information supported by the DOCUMENT EVIDENCE or MEMORY below.
- Cite claims supported by document evidence using [SOURCE <id>].
- Cite claims supported by memory using [MEMORY <id>].
- If neither the document evidence nor memory establishes part of the answer,
  explicitly say that the information is insufficient.
- Do not infer or invent missing company policy.

USER QUESTION:
{user_query}

DOCUMENT EVIDENCE:
{evidence}

MEMORY:
{memory}
"""
    client = get_client()
    response = client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=prompt,
        config=types.GenerateContentConfig(
            automatic_function_calling=
                types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
        ),
    )

    return response.text