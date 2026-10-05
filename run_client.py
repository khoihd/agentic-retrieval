from build_context import generate_grounded_answer, build_retrieval_context
from get_tools import get_agent_tools
from execute_function_call import execute_function_call
# from ollama import chat
from utils import get_client, MODEL, SYSTEM_PROMPT


def run_agent(
    user_query: str, retrieval_max_steps: int = 10, max_steps: int = 10
) -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    state = {
        "messages": messages,
        "step": 0,
        "max_steps": max_steps,
        "retrieval_step": 0,
        "retrieval_max_steps": retrieval_max_steps,
        "retrieved_chunks": [],
        "retrieved_memories": [],
        "retrieval_errors": [],
        "status": "running",
    }

    print(f"Running model: {MODEL}")
    print(f"=====Step {state['step']}=====")
    print(f"User query: {user_query}")

    agent_tools = get_agent_tools()
    while state["status"] == "running" and state["step"] < state["max_steps"]:
        response = get_client().chat.completions.create(
            model="qwen3:4b",
            messages=state["messages"],
            tools=agent_tools,
        )

        state["messages"].append(response.choices[0].message)

        state["step"] += 1

        # If there is no tool to run, return the text of the last response
        if not response.choices[0].message.tool_calls:
            state["status"] = "success"

            # No document or memory retrieval happened.
            if not state["retrieved_chunks"] and not state["retrieved_memories"]:
                return response.choices[0].message.content

            context_chunks, context_memories = build_retrieval_context(
                user_query, state["retrieved_chunks"], state["retrieved_memories"]
            )

            grounded_answer = generate_grounded_answer(
                user_query, context_chunks, context_memories
            )
            return grounded_answer

        print(f"=====Step {state['step']}=====")
        function_call_results = []

        for function_call in response.choices[0].message.tool_calls:
            if (
                function_call.function.name == "search_documents"
                and state["retrieval_step"] >= state["retrieval_max_steps"]
            ):
                result = {
                    "error": "retrieval_limit_reached",
                    "message": (
                        "Document retrieval limit reached. "
                        "No additional searches are allowed. "
                        "Use the evidence already retrieved to answer the user, "
                        "and clearly indicate anything that cannot be established "
                        "from the available evidence."
                    ),
                }
            else:
                try:
                    if function_call.function.name == "search_documents":
                        state["retrieval_step"] += 1

                    result = execute_function_call(function_call)

                    if function_call.function.name == "search_documents":
                        for chunk in result["result"]:
                            state["retrieved_chunks"].append(chunk)
                    elif function_call.function.name == "search_memory":
                        for memory in result["result"]:
                            state["retrieved_memories"].append(memory)
                except Exception as e:
                    error = {
                        "error": "retrieval_failed",
                        "message": str(e),
                    }
                    state["retrieval_errors"].append(error)
                    result = error

            print(f"tool_name: {function_call.function.name}")
            print(f"args: {function_call.function.arguments}")
            if "error" in result:
                print(result["error"])
                print(result["message"])

            function_call_results.append(
                {
                    "role": "tool",
                    "tool_call_id": function_call.id, 
                    "tool_name": function_call.function.name,
                    "content": str(result),
                }
            )

        state["messages"].extend(function_call_results)

    state["status"] = "failed"
    raise RuntimeError(f"Agent exceeded maximum number of steps ({max_steps})")
