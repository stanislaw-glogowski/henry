from dataclasses import dataclass

from langgraph.runtime import Runtime

from henry_common import Profile

type ReplyRuntime = Runtime[ReplyContext]


@dataclass(frozen=True, slots=True)
class ReplyContext:
    model: str = "ollama:gpt-oss:20b"
    name: str = "Henry"
    language: str = "English"
    system_prompt: str = (
        "You are {name}, a distinguished and impeccably mannered {language}-speaking "
        "voice assistant in the style of a classic personal butler."
    )

    @staticmethod
    def from_profile(profile: Profile) -> ReplyContext:
        return ReplyContext(
            model=profile.reply.model,
            language=profile.language,
            system_prompt=profile.reply.system_prompt,
        )
