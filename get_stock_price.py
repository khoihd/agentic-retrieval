from google.genai import types


def get_stock_price(symbol: str) -> dict:
    """Get the current stock price for a ticker symbol.
    Args:
        symbol: Stock ticker symbol, such as MSFT.
    """

    fake_prices = {
        "MSFT": 512.35,
        "AAPL": 230.10,
        "GOOG": 201.50,
    }

    symbol = symbol.upper()

    if symbol not in fake_prices:
        raise ValueError(f"Unknown stock symbol: {symbol}")

    return {
        "symbol": symbol,
        "price": fake_prices[symbol],
        "currency": "USD",
    }


def get_stock_price_declaration() -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name="get_stock_price",
        description="Get the current stock price for a stock ticker symbol.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol, for example AAPL or MSFT.",
                }
            },
            "required": ["symbol"],
        },
    )
