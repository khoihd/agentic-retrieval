from run_client import run_agent

def test_calling_memory():
    #response = run_agent("What Python version should I use for Project X?")
    # response = run_agent("What is 15 * 7?")
    response = run_agent("What is the maximum hotel reimbursement in Tokyo, and what Python version should I use for Project X?")
    print(response)

if __name__ == "__main__":
    test_calling_memory()