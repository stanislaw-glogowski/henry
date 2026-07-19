from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    voice_path: str
    system_prompt: str
