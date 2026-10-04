from get_tools import get_tool_registry


def execute_function_call(function_call):
    print(
        f"execute_function_call: {function_call.function.name} with {dict(function_call.function.arguments)}"
    )

    tool_name = function_call.function.name
    tool = get_tool_registry().get(tool_name)
    args = dict(function_call.function.arguments)

    if tool is None:
        return {
            "status": "error",
            "error": f"Unknown tool: {tool_name}",
        }

    try:
        result = tool(**args)
        return {
            "status": "success",
            "result": result,
        }
    except (TypeError, ValueError, KeyError, OSError) as e:
        return {
            "status": "error",
            "error": str(e),
        }
