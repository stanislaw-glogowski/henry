import re


class TurnEndpointDetector:
    """Estimate whether recognized text forms a complete conversational turn."""

    _TRAILING_CONNECTORS = frozenset(
        {
            "a",
            "ale",
            "albo",
            "and",
            "because",
            "bo",
            "chociaż",
            "czyli",
            "gdy",
            "i",
            "if",
            "jeśli",
            "kiedy",
            "który",
            "lub",
            "or",
            "ponieważ",
            "that",
            "więc",
            "więc jeśli",
            "when",
            "which",
            "że",
        }
    )
    _WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)

    def is_complete(self, text: str) -> bool:
        normalized = text.strip().lower()
        if not normalized:
            return False
        if normalized.endswith(
            (",", ";", ":", "-", "–", "—", "…", "...")  # noqa: RUF001
        ):
            return False
        words = self._WORDS.findall(normalized.rstrip(".!?"))
        if not words:
            return True
        endings = {words[-1], " ".join(words[-2:])}
        return self._TRAILING_CONNECTORS.isdisjoint(endings)
