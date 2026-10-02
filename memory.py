import json
import uuid
from datetime import datetime, timezone
from utils import embed_text, cosine_similarity, get_client
from google.genai import types
from MemoryRelationship import MemoryRelationship

MEMORY_FILE = "memories.json"


def rerank_memories(original_query, memories):
    query_embedding = embed_text(original_query)
    reranked_memories = []

    for memory in memories:
        memory_embedding = memory["embedding"]
        rerank_score = cosine_similarity(
            query_embedding,
            memory_embedding,
        )

        reranked_memory = memory.copy()
        reranked_memory["rerank_score"] = rerank_score
        reranked_memories.append(reranked_memory)

    reranked_memories.sort(
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    return reranked_memories


def search_memory(query):
    """
    Search persistent memory created from previous interactions.
    Use this to retrieve relevant user preferences, previous decisions,
    project-specific facts learned in earlier conversations, and other
    remembered information. This function searches memory, not shared
    company documents, policies, or other external knowledge sources.

    Args:
        query: A semantic search query describing the remembered
            information needed.

    Returns:
        A formatted string containing relevant retrieved memories, or a
        message indicating that no relevant memories were found.
    """
    memories = load_memories()
    results = find_related_memories(
        query,
        memories,
        top_k=1,
    )

    if not results:
        return "No relevant memories found."

    memories = []
    for result in results:
        memory = result["memory"]
        memories.append(
            {   "id": str(uuid.uuid4()),
                "text": memory["text"],
                "memory_type": memory["memory_type"],
                "scope": memory["scope"],
                "embedding": memory["embedding"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "active",
                "score": float(result["score"]),
            }
        )

    return memories


def format_memory_retrieval_context(retrieved_memories: list[dict]):
    formatted = []
    for memory in retrieved_memories:
        formatted.append(
            f"""MEMORY: {memory["id"]}
MEMORY TEXT: {memory["text"]}"""
        )

    return "\n\n".join(formatted)


# Assuming proposed_memory takes some form with three fields: text, memory_type, and scope
def process_new_memory(proposed_memory):
    # Load current memories
    existing_memories = load_memories()

    new_memory = create_memory(proposed_memory)

    related_memories = find_related_memories(
        new_memory["text"], existing_memories, top_k=3
    )
    if not related_memories:
        existing_memories.append(new_memory)
    else:
        candidates = [m["memory"] for m in related_memories]
        relationships = classify_memory_relationship(new_memory, candidates)
        memory_update(new_memory, existing_memories, relationships)

    # Write the updated memories to the file
    persist_memories(existing_memories)


def memory_update(new_memory, memories, relationships: list[MemoryRelationship]):
    # ["duplicate", "superseding", "conflicting", "related"]
    conflict = False
    superseding_ids = set()
    for r in relationships:
        if r.relationship == "duplicate":
            return
        elif r.relationship == "conflicting":
            conflict = True
        elif r.relationship == "superseding":
            superseding_ids.add(r.id)

    for m in memories:
        if m["id"] in superseding_ids:
            m["status"] = "superseded"

    if conflict:
        new_memory["status"] = "conflicted"
    else:
        new_memory["status"] = "active"

    memories.append(new_memory)


def persist_memories(memories):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)


def load_memories():
    existing_memory = []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            existing_memory = json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        existing_memory = []

    return existing_memory


def create_memory(memory):
    return {
        "id": str(uuid.uuid4()),
        "text": memory["text"],
        "memory_type": memory["memory_type"],
        "scope": memory["scope"],
        "embedding": embed_text(memory["text"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "active",
    }


def find_related_memories(query, memories, top_k=3):
    """Return a list of memory and score"""
    query_embedding = embed_text(query)
    results = []

    for memory in memories:
        if memory["status"] == "active":
            score = cosine_similarity(query_embedding, memory["embedding"])
            results.append({"memory": memory, "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def classify_memory_relationship(new_memory, existing_memories):
    prompt = f"""
    Find the semantic relationship between the new memory, namely NEW_MEMORY, and every existing memory, called EXISTING_MEMORY, in the collection EXISTING_MEMORIES
    Rules:
    - Return a list of MemoryRelationship, each representing the relationship between NEW_MEMORY and EXISTING_MEMORY
    - The attribute 'relationship' is defined as
        - 'duplicate': The new memory expresses essentially the same fact, preference, decision,
            or information as the existing memory. Minor differences in wording or
            additional non-material details do not make it a different memory.
        - 'superseding': The new memory describes a newer state, update, replacement, or change
            that makes the existing memory historical or outdated.
        - 'conflicting': The new and existing memories make incompatible claims, but there is not
            enough information to determine that the new memory represents a newer
            state that replaces the existing memory.
        - 'related': The new and existing memories concern the same or closely related subject,
            entity, project, or context, but they provide distinct information.
            Neither memory duplicates, supersedes, nor conflicts with the other.
    - The attribute 'id' is the id of the EXISTING_MEMORY
    - If none of the above relationship matches, then do not include the relationship MemoryRelationship between NEW_MEMORY and EXISTING_MEMORY in the result
    
    NEW_MEMORY:
    {new_memory}
    
    EXISTING_MEMORIES:
    {existing_memories}
    """

    client = get_client()
    response = client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[MemoryRelationship],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )

    return response.parsed


def get_search_memory_declaration() -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name="search_memory",
        description=(
            "Search persistent memory from previous interactions for relevant "
            "user preferences, project information, previous decisions, and "
            "other remembered information."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query describing the remembered information needed."
                    ),
                }
            },
            "required": ["query"],
        },
    )
