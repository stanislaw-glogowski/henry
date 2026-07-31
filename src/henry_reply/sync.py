import asyncio

from langgraph_sdk import get_client

from henry_reply.graph import ReplyContext, ReplyGraph
from henry_resources import LocalStore

CLIENT_URL = "http://localhost:2024"


async def run() -> None:
    local_store = LocalStore()

    client = get_client(url=CLIENT_URL)

    for profile in local_store.list_profiles():
        await client.assistants.create(
            ReplyGraph.NAME,
            name=profile.name,
            context=ReplyContext.from_profile(profile),
        )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
