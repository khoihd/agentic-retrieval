from utils import get_tool_registry


def execute_function_call(function_call):
    tool_name = function_call.name
    tool = get_tool_registry().get(tool_name)
    args = dict(function_call.args)

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