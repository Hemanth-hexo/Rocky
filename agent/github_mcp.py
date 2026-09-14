"""Bridges Rocky's tool loop to the user's existing GIT_MCP server (~/GIT_MCP,
a Node/stdio MCP server exposing search_github_repos and friends). Each call
spawns the server fresh over stdio, calls the one tool, and tears it down."""

import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

GIT_MCP_DIR = os.environ.get("GIT_MCP_PATH", os.path.expanduser("~/GIT_MCP"))


async def _call_search_github_repos(query: str, min_stars: int, language: str, limit: int) -> str:
    params = StdioServerParameters(command="node", args=["server.js"], cwd=GIT_MCP_DIR)
    filters = {}
    if min_stars:
        filters["min_stars"] = min_stars
    if language:
        filters["language"] = language
    if limit:
        filters["limit"] = limit

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            arguments = {"query": query}
            if filters:
                arguments["filters"] = filters
            result = await session.call_tool("search_github_repos", arguments)
            parts = [block.text for block in result.content if hasattr(block, "text")]
            return "\n".join(parts) if parts else "(no results)"


def search_github_repos(query: str, min_stars: int = 0, language: str = "", limit: int = 10) -> str:
    return asyncio.run(_call_search_github_repos(query, min_stars, language, limit))


GITHUB_FUNCTIONS = {"search_github_repos": search_github_repos}

GITHUB_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_github_repos",
            "description": (
                "Search GitHub for open-source repositories relevant to a research topic, "
                "technology, or project idea. Returns the top matches ranked by stars and "
                "recent activity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for, phrased as GitHub search terms, e.g. 'retrieval augmented generation vector database'.",
                    },
                    "min_stars": {"type": "integer", "description": "Minimum star count (optional)."},
                    "language": {"type": "string", "description": "Filter to a programming language (optional)."},
                    "limit": {"type": "integer", "description": "Max results to return (optional, default 10)."},
                },
                "required": ["query"],
            },
        },
    },
]
