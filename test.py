from memory import (
    search_memory,
    load_memories,
    find_related_memories,
    persist_memories,
    create_memory,
    classify_memory_relationship,
    process_new_memory,
    extract_memories,
    write_memories
)
from run_client import run_agent
from ollama import chat
from utils import MODEL

def test_memory_context():
    user_query = "What Python version should I use for Project X?"
    memory_context = search_memory(user_query)

    prompt = f"""
    Answer the user's question using the relevant memories when useful.
    RELEVANT MEMORIES:
    {memory_context}

    USER:
    {user_query}
    """

    response = chat(
        model=MODEL,
        messages=prompt,
        think=False,
    )

    print(response.message.content)


def test_persist_memory():
    proposed_memories = [
        {
            "text": "User prefers pytest for personal projects.",
            "memory_type": "preference",
            "scope": "user",
        },
        {
            "text": "Project X uses PostgreSQL.",
            "memory_type": "project_fact",
            "scope": "project_x",
        },
        {
            "text": "User prefers concise technical explanations.",
            "memory_type": "preference",
            "scope": "user",
        },
    ]
    memories = load_memories()
    for memory in proposed_memories:
        memories.append(create_memory(memory))

    persist_memories(memories)
    results = find_related_memories(
        "What database does Project X use?",
        load_memories(),
        top_k=3,
    )

    for result in results:
        print(result["score"], result["memory"]["text"])


def classify_memory():
    existing_memories = [
        {
            "id": "m1",
            "text": "Project Alpha uses Redis for caching.",
            "memory_type": "project_fact",
            "scope": "project_alpha",
            "status": "active",
        },
        {
            "id": "m2",
            "text": "Project Alpha runs on Python 3.12.",
            "memory_type": "project_fact",
            "scope": "project_alpha",
            "status": "active",
        },
        {
            "id": "m3",
            "text": "Project Alpha uses PostgreSQL as its primary database.",
            "memory_type": "project_fact",
            "scope": "project_alpha",
            "status": "active",
        },
        {
            "id": "m4",
            "text": "Project Alpha uses FastAPI for its web API.",
            "memory_type": "project_fact",
            "scope": "project_alpha",
            "status": "active",
        },
        {
            "id": "m5",
            "text": "Project Alpha's caching layer is Redis.",
            "memory_type": "project_fact",
            "scope": "project_alpha",
            "status": "active",
        },
    ]

    new_memory = {
        "text": (
            "Project Alpha migrated its caching layer from Redis to Memcached. "
            "It currently runs on Python 3.13. "
            "Its primary database is MySQL. "
            "FastAPI is used for the web API."
        ),
        "memory_type": "project_fact",
        "scope": "project_alpha",
    }

    relationship_result = classify_memory_relationship(new_memory, existing_memories)
    print(relationship_result.parsed)
    for relationship in relationship_result.parsed:
        print(relationship)


def test_memory():
    memory = {
        "text": "Project X requires Python 3.12.",
        "memory_type": "project_fact",
        "scope": "project_alpha",
    }
    process_new_memory(memory)

    memory = {
        "text": "Project X now requires Python 3.13.",
        "memory_type": "project_fact",
        "scope": "project_alpha",
    }
    process_new_memory(memory)

    memory = {
        "text": "User prefers pytest.",
        "memory_type": "project_fact",
        "scope": "project_alpha",
    }
    process_new_memory(memory)

    memory = {
        "text": "Project X uses PostgreSQL.",
        "memory_type": "project_fact",
        "scope": "project_alpha",
    }
    process_new_memory(memory)
    memories = load_memories()
    for m in memories:
        print(m["text"], m["status"])


def test_document_query():
    original_query = "I'm traveling to Tokyo for work. " + \
        "How much can I spend on my hotel and meals, " + \
        "and what receipts do I need to keep?"

    result = run_agent(original_query)
    print(result)

def test_memory_extraction():
    query = "Project X uses PostgreSQL." \
            "What's 15 * 7?" \
            "Thanks!" \
            "I prefer pytest for my personal projects." \
            "I'm debugging a PostgreSQL error right now." \
            "Project X used Python 3.12 last year but now requires 3.13."
    extracted_memories = extract_memories(query)
    for extracted in extracted_memories:
        print(extracted)

def test_query():
    query = "Project X uses PostgreSQL." \
        "What's 15 * 7?" \
        "Thanks!" \
        "I prefer pytest for my personal projects." \
        "I'm debugging a PostgreSQL error right now." \
        "Project X used Python 3.12 last year but now requires 3.13."
    
    run_agent(query)

def test_process_new_memory():
    query = "Project X uses PostgreSQL." \
            "What's 15 * 7?" \
            "Thanks!" \
            "I prefer pytest for my personal projects." \
            "I'm debugging a PostgreSQL error right now." \
            "Project X used Python 3.12 last year but now requires 3.13."
    process_new_memory(query)

def test_write_memories():
    queries = ["Project X requires Python 3.13.",
               "Project X uses PostgreSQL for its primary database.",
               "Project X uses FastAPI."]

    write_memories(queries)
    
if __name__ == "__main__":
    test_write_memories()