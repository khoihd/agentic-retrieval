from google.genai import types
from utils import get_client
from build_context import generate_grounded_answer, build_retrieval_context
from get_tools import get_agent_tools
from execute_function_call import execute_function_call

def run_agent(user_query: str, retrieval_max_steps: int = 3, max_steps: int = 10) -> None:
    contents = [types.Content(role="user", parts=[types.Part(text=user_query)])]

    state = {
        "contents": contents,
        "step": 0,
        "max_steps": max_steps,
        "retrieval_step": 0,
        "retrieval_max_steps": retrieval_max_steps,
        "retrieved_chunks": [],
        "retrieved_memories": [],
        "retrieval_errors": [],
        "status": "running",
    }

    print(f"=====Step {state['step']}=====")
    print(f"User query: {user_query}")

    client = get_client()
    agent_tools = get_agent_tools()
    while state["status"] == "running" and state["step"] < state["max_steps"]:
        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=state["contents"],
            config=types.GenerateContentConfig(
                tools=[agent_tools],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )

        state["step"] += 1
        model_content = response.candidates[0].content
        state["contents"].append(model_content)

        function_calls = [
            part.function_call
            for part in model_content.parts
            if part.function_call is not None
        ]

        # If there is no tool to run, return the text of the last response
        if not function_calls:
            state["status"] = "success"

            # No document or memory retrieval happened.
            if not state["retrieved_chunks"] and not state["retrieved_memories"]:
                return response.text

            context_chunks, context_memories = build_retrieval_context(
                user_query,
                state["retrieved_chunks"],
                state["retrieved_memories"]
            )

            grounded_answer = generate_grounded_answer(user_query, context_chunks, context_memories)
            return grounded_answer

        print(f"=====Step {state['step']}=====")

        function_response_parts = []

        for function_call in function_calls:
            if function_call.name == "search_documents" and state["retrieval_step"] >= state["retrieval_max_steps"]:
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
                print(f"result: {result}")
            else:
                try:
                    if function_call.name == "search_documents":
                        state["retrieval_step"] += 1

                    result = execute_function_call(function_call)

                    if function_call.name == "search_documents":
                        for chunk in result["result"]:
                            state['retrieved_chunks'].append(chunk)
                    elif function_call.name == "search_memory":
                        for memory in result["result"]:
                            state['retrieved_memories'].append(memory)
                except Exception as e:
                    error = {
                        "error": "retrieval_failed",
                        "message": str(e),
                    }
                    state["retrieval_errors"].append(error)
                    result = error

            print(f"tool_name: {function_call.name}")
            print(f"args: {function_call.args}")
            for r in result['result']:
                for key, value in r.items():
                    if key != "embedding":
                        print(key, value)

            function_response_parts.append(
                types.Part.from_function_response(
                    name=function_call.name,
                    response={"result": result},
                )
            )

        state["contents"].append(
            types.Content(role="user", parts=function_response_parts)
        )

    state["status"] = "failed"
    raise RuntimeError(f"Agent exceeded maximum number of steps ({max_steps})")