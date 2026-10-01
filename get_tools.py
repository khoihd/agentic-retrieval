from get_stock_price import get_stock_price, get_stock_price_declaration
from search_documents import search_documents, get_search_documents_declaration
from memory import search_memory, get_search_memory_declaration
from google.genai import types

def get_agent_tools() -> types.Tool:
    return types.Tool(
    function_declarations=[
        get_stock_price_declaration(),
        get_search_documents_declaration(),
        get_search_memory_declaration(),
    ]
)

def get_tool_registry() -> dict:
    return {
    "get_stock_price": get_stock_price,
    "search_documents": search_documents,
    "search_memory": search_memory,
}