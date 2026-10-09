from memory import (
    MemoryReference,
    classify_memory_relationship,
    create_memory,
    extract_memories,
    find_related_memories,
    load_memories,
    persist_memories,
    process_new_memory,
    search_memory,
    write_memories,
)
from run_client import run_agent
from utils import MODEL, get_client


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

    response = get_client().chat.completions.create(
        model=MODEL,
        messages=prompt,
        temperature=0,
        extra_body={"think": False},
    )

    print(response.choices[0].message.content)


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
    original_query = (
        "I'm traveling to Tokyo for work. "
        + "How much can I spend on my hotel and meals, "
        + "and what receipts do I need to keep?"
    )

    result = run_agent(original_query)
    print(result)


def test_memory_extraction():
    # query = (
    #     "Project X uses PostgreSQL."
    #     "What's 15 * 7?"
    #     "Thanks!"
    #     "I prefer pytest for my personal projects."
    #     "I'm debugging a PostgreSQL error right now."
    #     "Project X used Python 3.12 last year but now requires 3.13."
    # )

    query = "Project X requires Python 3.14 and uses Redis for caching"

    extracted_memories = extract_memories(query)
    for extracted in extracted_memories:
        print(extracted)


def test_query():
    query = (
        "Project X uses PostgreSQL."
        "What's 15 * 7?"
        "Thanks!"
        "I prefer pytest for my personal projects."
        "I'm debugging a PostgreSQL error right now."
        "Project X used Python 3.12 last year but now requires 3.13."
    )

    run_agent(query)


def test_process_new_memory():
    query = (
        "Project X uses PostgreSQL."
        "What's 15 * 7?"
        "Thanks!"
        "I prefer pytest for my personal projects."
        "I'm debugging a PostgreSQL error right now."
        "Project X used Python 3.12 last year but now requires 3.13."
    )
    process_new_memory(query)


def test_end_to_end_memory_process():
    queries = ["Project X requires Python 3.13",
                "Project X uses PostgreSQL as its primary database",
                "Project X uses FastAPI."
            ]

    write_memories(queries)

    process_new_memory("Project X now requires Python 3.14 and uses Redis for caching.")

    memories = load_memories()
    for memory in memories:
        print(MemoryReference.from_memory(memory))


def test_token_consumption():
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": 'Return exactly this JSON: {"memories": []}',
            }
        ],
        temperature=0,
        max_tokens=256,
        extra_body={"think": False},
    )
    print("Finish reason:", response.choices[0].finish_reason)
    print("Content:", repr(response.choices[0].message.content))
    print("Usage:", response.usage)

def test_duplicate_memory():
    queries = ["Project X uses PostgreSQL."]
    write_memories(queries)
    
    process_new_memory("Project X uses PostgreSQL.")

    memories = load_memories()
    for memory in memories:
        print(MemoryReference.from_memory(memory))

def test_conflicting_memory():
    queries = ["Project X uses PostgreSQL as its primary database."]
    write_memories(queries)
    
    process_new_memory("Project X uses MySQL as its primary database.")

    memories = load_memories()
    for memory in memories:
        print(MemoryReference.from_memory(memory))


def test_false_memory_extraction():
    memories = extract_memories("Should Project X migrate from PostgreSQL to MySQL?")
    print(memories)

def test_missed_memory_extraction():
    query = (
        "By the way, for all my future Python projects, "
        "please use pytest instead of unittest. "
        "Anyway, what's 15 * 7?"
    )

    memories = extract_memories(query)
    for memory in memories:
        print(memory)

def test_semantic():
    queries = ["The user prefers pytest instead of unittest for Python projects."]
    write_memories(queries)
    search_query = "Which testing framework should I use for my next Python application?"
    memories = search_memory(query=search_query, scope="user", top_k=3)
    
    for memory in memories:
        print(MemoryReference.from_memory(memory))

def test_filtering():
    queries = ["Project X uses PostgreSQL.",
               "Project Y uses MySQL.",
               "The user prefers pytest."
    ]

    print("Existing memories:")
    for memory in load_memories():
        print(MemoryReference.from_memory(memory))

    write_memories(queries)
    query = "What database does the project use?"

    memories = search_memory(query=query, scope="project_x", top_k=10)

    print("Searching for memory:")
    for memory in memories:
        print(MemoryReference.from_memory(memory))

    assert len(memories) == 1
    assert memories[0].scope == "project_x"
    assert memories[0].text == "Project X uses PostgreSQL."


def test_top_k_retrieval_recall():
    queries = ["The user prefers pytest for Python testing.", 
        "The user prefers Ruff for Python linting.",
        "The user prefers Black for Python formatting.",
        "The user prefers VS Code for Python development.",
        "The user prefers PostgreSQL for personal projects."
    ]

    write_memories(queries)
    query = "What are my preferences for Python development tools?"
    for k in [1, 3, 5]:
        memories = search_memory(query=query, scope="user", top_k=k)

        print(f"\nTop {k} memories:")

        for memory in memories:
            print(f"- {MemoryReference.from_memory(memory)}")

        assert len(memories) <= k


if __name__ == "__main__":
    test_top_k_retrieval_recall()
