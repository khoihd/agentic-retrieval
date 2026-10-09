from memory import create_memory, persist_memories
from run_client import run_agent


def test_calling_memory():
    # query = "What Python version should I use for Project X?"
    # query = "What is 15 * 7?"
    # query = "What is the maximum hotel reimbursement in Tokyo, and what Python version should I use for Project X?"
    query = "Tell me everything you remember about the technologies used by Project X."
    response = run_agent(query)
    print(response)


def test_failing_memory_retrieval():
    test_memories = [
        {
            "text": "Project X uses PostgreSQL for its primary database.",
            "memory_type": "project_fact",
            "scope": "project_x",
        },
        {
            "text": "Project X uses Redis for caching.",
            "memory_type": "project_fact",
            "scope": "project_x",
        },
        {
            "text": "Project X uses FastAPI for its API layer.",
            "memory_type": "project_fact",
            "scope": "project_x",
        },
        {
            "text": "Project X uses Docker for local development.",
            "memory_type": "project_fact",
            "scope": "project_x",
        },
        {
            "text": "Project X requires Python 3.13.",
            "memory_type": "project_fact",
            "scope": "project_x",
        },
    ]

    created_memories = [create_memory(m) for m in test_memories]
    persist_memories(created_memories)

    query = "Tell me everything you remember about the technologies used by Project X."
    response = run_agent(query)
    print(response)
    


if __name__ == "__main__":
    test_failing_memory_retrieval()
