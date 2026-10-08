import json
import uuid
from datetime import datetime, timezone
from typing import Literal

from utils import embed_text, cosine_similarity, MODEL, MAX_MEMORY_RETRIEVAL, MEMORY_RELATIONSHIP_TOP_K, DEFAULT_MEMORY_SEARCH_TOP_K
from MemoryRelationship import MemoryRelationship, MemoryRelationshipList
from pydantic import BaseModel
from utils import get_client


MEMORY_FILE = "memories.json"


class Memory(BaseModel):
    id: str
    text: str
    memory_type: str
    scope: str
    embedding: list[float]
    created_at: str
    status: Literal["active", "superseded", "conflicted"]


class ExtractedMemory(BaseModel):
    text: str
    memory_type: str
    scope: str


class ExtractedMemoryList(BaseModel):
    memories: list[ExtractedMemory]


def extract_memories(text: str) -> list[ExtractedMemory]:
    prompt = f"""

Extract information from the input that is worth storing in long-term memory.

Store information that:
- is likely to be useful in future interactions
- is reasonably durable rather than temporary
- describes user preferences, decisions, or project-specific facts
- can be represented as a clear, self-contained fact

Do not store:
- conversational filler
- questions that contain no useful durable information
- temporary task state that is only useful in the current interaction
- information unlikely to matter in future interactions

Guidelines:
- Each memory should contain one atomic fact when possible.
- Rewrite memories as clear, self-contained statements.
- A single input may produce zero, one, or multiple memories.
- Choose an appropriate memory_type.
- Choose an appropriate scope.

Scope rules:
- Use "user" for information about the user's general preferences,
  habits, or personal information.
- For project-specific information, derive a normalized project scope.
  For example:
  "Project X" -> "project_x"
  "Project Alpha" -> "project_alpha"
- Do not use "global" for user-specific or project-specific information.

INPUT:
{text}
"""
    response = get_client().beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        response_format=ExtractedMemoryList,
        temperature=0,
    )

    extracted_memory_list = response.choices[0].message.parsed

    return extracted_memory_list.memories


def rerank_memories(original_query: str, memories: list[Memory]) -> list[Memory]:
    query_embedding = embed_text(original_query)
    reranked_memories = []

    for memory in memories:
        memory_embedding = memory.embedding
        rerank_score = cosine_similarity(
            query_embedding,
            memory_embedding,
        )

        reranked_memory = {"memory": memory, "rerank_score": rerank_score}
        reranked_memories.append(reranked_memory)

    reranked_memories.sort(
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    return [entry["memory"] for entry in reranked_memories]


def search_memory(query: str, scope: str = "", top_k: int = DEFAULT_MEMORY_SEARCH_TOP_K) -> list[Memory]:
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
    top_k = min(top_k, MAX_MEMORY_RETRIEVAL)
    memories = load_memories()
    related_memories = find_related_memories(query, memories, scope, top_k)

    return related_memories


def format_memory_retrieval_context(retrieved_memories: list[Memory]) -> str:
    formatted = []
    for memory in retrieved_memories:
        formatted.append(
            f"""MEMORY: {memory.id}
MEMORY TEXT: {memory.text}"""
        )

    return "\n\n".join(formatted)


# Assuming proposed_memory takes some form with three fields: text, memory_type, and scope
def process_new_memory(user_query: str) -> None:
    # Load current memories
    existing_memories = load_memories()
    
    # Extract key memories from user_query
    proposed_memories = extract_memories(user_query)
    
    for proposed_memory in proposed_memories:
        new_memory = create_memory(proposed_memory)

        related_memories = find_related_memories(new_memory.text, existing_memories, new_memory.scope)
        
        if not related_memories:
            existing_memories.append(new_memory)
        else:
            relationships = classify_memory_relationship(new_memory, related_memories)
            memory_update(new_memory, existing_memories, relationships)

    # Write the updated memories to the file
    persist_memories(existing_memories)


def memory_update(new_memory: Memory, memories: list[Memory], relationships: list[MemoryRelationship]) -> None:
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
        if m.id in superseding_ids:
            m.status = "superseded"

    if conflict:
        new_memory.status = "conflicted"
    else:
        new_memory.status = "active"

    memories.append(new_memory)


def persist_memories(memories: list[Memory]) -> None:
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump([memory.model_dump() for memory in memories], 
                  f, indent=2, ensure_ascii=False)


def load_memories() -> list[Memory]:
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return [Memory.model_validate(item) for item in data]
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def create_memory(memory: ExtractedMemory) -> Memory:
    return Memory(
        id=str(uuid.uuid4()),
        text=memory.text,
        memory_type=memory.memory_type,
        scope=memory.scope,
        embedding=embed_text(memory.text),
        created_at=datetime.now(timezone.utc).isoformat(),
        status="active",
    )


def write_memories(queries: list[str]) -> None:
    persist_memories([])

    new_memories = []
    extracted_memories = extract_memories(".".join(queries))
    for extracted_memory in extracted_memories:
        new_memories.append(create_memory(extracted_memory))

    persist_memories(new_memories)


def find_related_memories(query: str, memories: list[Memory], scope: str = None, top_k: int = MEMORY_RELATIONSHIP_TOP_K) -> list[Memory]:
    """Return a list of memory and score"""
    query_embedding = embed_text(query)
    results = []

    for memory in memories:
        if memory.status != "active":
            continue

        if not scope or memory.scope == scope:
            score = cosine_similarity(query_embedding, memory.embedding)
            results.append({"memory": memory, "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    return [entry["memory"] for entry in results[:top_k]]


def classify_memory_relationship(new_memory: Memory, existing_memories: list[Memory]):
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

    response = get_client().chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=MemoryRelationshipList,
    )

    return response.choices[0].message.parsed.relationships


def get_search_memory_declaration() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search persistent memory from previous interactions for "
                "user preferences, previous decisions, project-specific facts, "
                "and other remembered information. Do not use this to search "
                "shared company documents or policies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query describing the remembered information needed."
                        ),
                    },
                    "scope": {
                        "type": "string",
                        "description": (
                            "Optional scope used to restrict the memory search, "
                            "such as 'project_x'. Omit when no specific scope is known."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": (
                            "Maximum number of relevant memories to retrieve. "
                            "Use a small value for specific questions and a larger "
                            "value for broad questions that require more complete coverage."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    }
