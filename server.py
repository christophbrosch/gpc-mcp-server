# server.py
from fastmcp import FastMCP
import requests
import extruct
from starlette.requests import Request
from starlette.responses import PlainTextResponse

import torch

from gpc.types import Brick
from gpc.gpc_brick import (
    get_brick_class,
    search_bricks,
)
from gpc.gpc_class import (
    get_class_bricks,
)

# Create an MCP server
mcp = FastMCP(
    "GPC Service",
    dependencies=[
        "requests",
        "torch",
        "transformers",
        "extruct",
        "git+https://gitlab.rlp.net/KE3P/gpc@english-food-only",
    ],
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@mcp.tool("get_jsonld_product_data", description="get jsonld product data")
def get_jsonld_product_data(url: str) -> dict:
    """Get jsonld product data."""
    response = requests.get(url, timeout=10)
    return extruct.extract(response.text, syntaxes=["json-ld"])["json-ld"][
        0
    ]  # Hardcoded for MyTime


@mcp.tool(
    "search_for_bricks",
    description="search for bricks, consider the query string as keywords. Duplicates are removed.",
)
def search_for_bricks(query: str) -> list[Brick]:
    """Search for bricks."""
    return search_bricks(query)


@mcp.tool("get_brick_sibling", description="get all siblings of a brick")
def get_bricks_for_class(brick_code: str | int) -> list[Brick]:
    """Get all bricks for a given class."""
    class_code = get_brick_class(brick_code).class_code
    return get_class_bricks(class_code)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0")
