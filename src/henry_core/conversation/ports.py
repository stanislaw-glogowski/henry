from collections.abc import Generator


class MessageChunk:
    pass


class Message:
    pass


class LLModel:
    def generate(self) -> Generator[MessageChunk]:
        pass
