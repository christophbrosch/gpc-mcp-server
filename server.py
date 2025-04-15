# server.py
from fastmcp import FastMCP
import requests
import extruct

import torch

from gpc.types import Brick, Class, Family, Segment
from gpc.gpc_brick import (
    get_brick,
    get_brick_class,
    get_brick_family,
    get_brick_segment,
    search_bricks,
    search_bricks_class,
    search_bricks_family,
)
from gpc.gpc_class import (
    get_class,
    get_class_bricks,
    search_classes,
    search_classes_family,
)
from gpc.gpc_family import get_family, get_family_classes
from gpc.gpc_segment import get_segment, get_segments, get_segment_families

# Create an MCP server
mcp = FastMCP(
    "GPC Service",
    dependencies=[
        "requests",
        "torch",
        "transformers",
        "extruct",
        "git+https://gitlab.rlp.net/KE3P/gpc@english",
    ],
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@mcp.tool("get_brick_object", description="get information regarding one gpc brick")
def get_brick_object(brick_code: str) -> Brick:
    """Get information regarding a GPC brick code."""
    return get_brick(brick_code)


@mcp.tool("get_class_code_for_brick", description="get class code for a given brick")
def get_class_code_for_brick(brick_code: str) -> str:
    """Get the class code for a given brick code."""
    return get_brick_class(brick_code)


@mcp.tool("get_family_code_for_brick", description="get family code for a given brick")
def get_family_code_for_brick(brick_code: str) -> str:
    """Get the family code for a given brick code."""
    return get_brick_family(brick_code)


@mcp.tool(
    "get_segment_code_for_brick", description="get segment code for a given brick"
)
def get_segment_code_for_brick(brick_code: str) -> str:
    """Get the segment code for a given brick code."""
    return get_brick_segment(brick_code)


def get_class_object_for_brick(brick_code: str) -> Class:
    """Get information regarding a GPC class code."""
    return get_class(get_brick(brick_code).class_code)


@mcp.tool("get_bricks_for_class", description="get all bricks for a given class")
def get_bricks_for_class(class_code: str) -> list[Brick]:
    """Get all bricks for a given class."""
    return get_class_bricks(class_code)


@mcp.tool("get_class_object", description="get information regarding one gpc class")
def get_class_object(class_code: str) -> Class:
    """Get information regarding a GPC class code."""
    return get_class(class_code)


@mcp.tool("get_classes_for_family", description="get all classes for a given family")
def get_classes_for_family(family_code: str) -> list[Class]:
    """Get all classes for a given family."""
    return get_family_classes(family_code)


@mcp.tool("get_family_object", description="get information regarding one gpc family")
def get_family_object(family_code: str) -> Family:
    """Get information regarding a GPC family code."""
    return get_family(family_code)


@mcp.tool(
    "get_families_for_segment", description="get all families for a given segment"
)
def get_family_objects_for_segment(segment_code: str) -> list[Family]:
    """Get all families for a given segment."""
    return get_segment_families(segment_code)


@mcp.tool("get_segment_object", description="get information regarding one gpc segment")
def get_segment_object(segment_code: str) -> Segment:
    """Get information regarding a GPC segment code."""
    return get_segment(segment_code)


@mcp.tool("get_segments", description="get all segments")
def get_segment_objects() -> list[Segment]:
    """Get all segments."""
    return get_segments()


@mcp.tool("get_jsonld_product_data", description="get jsonld product data")
def get_jsonld_product_data(url: str) -> dict:
    """Get jsonld product data."""
    response = requests.get(url, timeout=10)
    return extruct.extract(response.text, syntaxes=["json-ld"])["json-ld"][
        0
    ]  # Hardcoded for MyTime


@mcp.tool("search_for_bricks", description="search for bricks")
def search_for_bricks(query: str) -> list[Brick]:
    """Search for bricks."""
    return search_bricks(query)


@mcp.tool("search_for_bricks_in_class", description="search for bricks in a class")
def search_for_bricks_in_class(query: str, class_code: str) -> list[Brick]:
    """Search for bricks in a class."""
    return search_bricks_class(query, class_code)


@mcp.tool("search_for_bricks_in_family", description="search for bricks in a family")
def search_for_bricks_in_family(query: str, family_code: str) -> list[Brick]:
    """Search for bricks in a family."""
    return search_bricks_family(query, family_code)


@mcp.tool("search_for_classes", description="search for classes")
def search_for_classes(query: str) -> list[Class]:
    """Search for classes."""
    return search_classes(query)


@mcp.tool("search_for_classes_in_family", description="search for classes in a family")
def search_for_classes_in_family(query: str, family_code: str) -> list[Class]:
    """Search for classes in a family."""
    return search_classes_family(query, family_code)


# class ProductData(BaseModel):
#     """Product data model."""

#     name: str = Field(description="Product name")
#     description: str = Field(description="Product description")
#     brand: str = Field(description="Brand name")
#     category: Optional[str] = Field(None, description="Product category")
#     image: Optional[str] = Field(None, description="Product image URL")
#     url: str = Field(description="Product URL")


# class_prediction_pipeline = pipeline(
#     "text-classification",
#     model=os.path.dirname(__file__) + "/models/class/",
#     device=DEVICE,
# )


# @mcp.tool(
#     "get_classification_class_code",
#     description="get class code predictions for a given product",
# )
# def get_classification_class_code(product_data: ProductData) -> list[str]:
#     """Get class code predictions for a given product. Which might be wrong."""
#     # Get the class code predictions
#     text = f"{product_data.url} ### {product_data.name} ### {product_data.description} "
#     predictions = class_prediction_pipeline(
#         text, top_k=5, truncation=True, max_length=512
#     )
#     predictions = [prediction["label"] for prediction in predictions]
#     return predictions
