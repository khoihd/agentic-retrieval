from run_client import run_agent

def test_calling_memory():
    # query = "What Python version should I use for Project X?"
    # query = "What is 15 * 7?"
    # query = "What is the maximum hotel reimbursement in Tokyo, and what Python version should I use for Project X?"
    query = "Tell me everything you remember about the technologies used by Project X."
    response = run_agent(query)
    print(response)

if __name__ == "__main__":
    test_calling_memory()