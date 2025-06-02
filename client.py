import asyncio
from fastmcp import Client

client = Client("http://0.0.0.0:8000/sse/")


async def main():
    async with client:
        print(f"Client connected: {client.is_connected()}")
        response = await client.call_tool("search_for_bricks", {"query": "chocolate"})
        print(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
