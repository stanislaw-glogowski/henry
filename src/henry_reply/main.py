import asyncio

from langgraph_sdk import get_client

from henry_reply.reply import ReplyContext, ReplyGraph
from henry_common.profiles import load_profiles

CLIENT_URL = "http://localhost:2024"


async def run() -> None:
    profiles = load_profiles()

    client = get_client(url=CLIENT_URL)

    for profile in profiles.values():
        await client.assistants.create(
            ReplyGraph.NAME,
            name=profile.name,
            context=ReplyContext.from_profile(profile),
        )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
