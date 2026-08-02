import re


class ReplySegmenter:
    """Turn streamed model text into complete, speakable phrases."""

    _ABBREVIATIONS = frozenset(
        {
            "dr.",
            "hab.",
            "in.",
            "itd.",
            "itp.",
            "m.",
            "mgr.",
            "min.",
            "np.",
            "prof.",
            "r.",
            "tj.",
            "tzn.",
            "ul.",
        }
    )
    _CLOSING_PUNCTUATION = frozenset("\"'”’)]}")  # noqa: RUF001
    _INITIALS = re.compile(r"(?:[^\W\d_]\.){2,}$", re.UNICODE)

    def __init__(self, *, soft_limit: int = 72, hard_limit: int = 160) -> None:
        if soft_limit <= 0 or hard_limit < soft_limit:
            raise ValueError(
                "Reply phrase limits must satisfy 0 < soft_limit <= hard_limit; "
                f"got soft_limit={soft_limit}, hard_limit={hard_limit}"
            )
        self._soft_limit = soft_limit
        self._hard_limit = hard_limit
        self._buffer = ""

    def feed(self, text: str) -> tuple[str, ...]:
        self._buffer += text.replace("\r\n", "\n").replace("\r", "\n")
        return self._extract()

    def flush(self) -> tuple[str, ...]:
        phrases = list(self._extract())
        if phrase := self._buffer.strip():
            phrases.append(phrase)
        self._buffer = ""
        return tuple(phrases)

    def _extract(self) -> tuple[str, ...]:
        phrases: list[str] = []
        while (boundary := self._find_boundary()) is not None:
            phrase = self._buffer[:boundary].strip()
            self._buffer = self._buffer[boundary:].lstrip()
            if phrase:
                phrases.append(phrase)
        return tuple(phrases)

    def _find_boundary(self) -> int | None:
        for index, character in enumerate(self._buffer):
            if character == "\n":
                return index + 1

            if character in ".!?…" and not self._is_protected_period(index):
                if boundary := self._punctuation_boundary(index):
                    return boundary

            if character in ",;:" and index + 1 >= self._soft_limit:
                if self._followed_by_space(index):
                    return index + 1

            if index + 1 >= self._hard_limit and character.isspace():
                return index + 1

        return None

    def _punctuation_boundary(self, index: int) -> int | None:
        boundary = index + 1
        while boundary < len(self._buffer) and self._buffer[boundary] in ".!?…":
            boundary += 1
        while (
            boundary < len(self._buffer)
            and self._buffer[boundary] in self._CLOSING_PUNCTUATION
        ):
            boundary += 1
        if boundary == len(self._buffer) or self._buffer[boundary].isspace():
            return boundary
        return None

    def _is_protected_period(self, index: int) -> bool:
        if self._buffer[index] != ".":
            return False
        if (
            index > 0
            and index + 1 < len(self._buffer)
            and self._buffer[index - 1].isdigit()
            and self._buffer[index + 1].isdigit()
        ):
            return True
        if (
            index + 2 < len(self._buffer)
            and self._buffer[index + 1].isalpha()
            and self._buffer[index + 2] == "."
        ):
            return True

        start = index
        while start > 0 and not self._buffer[start - 1].isspace():
            start -= 1
        token = (
            self._buffer[start : index + 1]
            .lower()
            .strip(
                "\"'“‘([{"  # noqa: RUF001
            )
        )
        return token in self._ABBREVIATIONS or bool(self._INITIALS.fullmatch(token))

    def _followed_by_space(self, index: int) -> bool:
        return index + 1 == len(self._buffer) or self._buffer[index + 1].isspace()
