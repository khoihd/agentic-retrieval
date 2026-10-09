from get_stock_price import get_stock_price, get_stock_price_declaration
from memory import get_search_memory_declaration, search_memory
from search_documents import get_search_documents_declaration, search_documents


def get_agent_tools() -> list:
    return [
        get_stock_price_declaration(),
        get_search_documents_declaration(),
        get_search_memory_declaration(),
    ]


def get_tool_registry() -> dict:
    return {
        "get_stock_price": get_stock_price,
        "search_documents": search_documents,
        "search_memory": search_memory,
    }
