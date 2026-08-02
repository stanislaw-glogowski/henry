from .domain import ResponseMode, ResponsePlan, TurnIntent


class ResponseRouter:
    """Select response depth without adding another model call to obvious turns."""

    CLASSIFICATION_PROMPT = (
        "Classify the requested response depth. Return FAST for a short direct "
        "answer or DETAILED for a multi-step, explanatory, creative, or "
        "comprehensive answer. Return only FAST or DETAILED."
    )
    _FAST_WORD_LIMIT = 14
    _DETAILED_WORD_LIMIT = 24

    def is_ambiguous(self, text: str) -> bool:
        word_count = len(text.split())
        if self._FAST_WORD_LIMIT < word_count < self._DETAILED_WORD_LIMIT:
            return True
        return 3 < word_count <= self._FAST_WORD_LIMIT and not text.rstrip().endswith(
            "?"
        )

    def classified_plan(self, classification: str) -> ResponsePlan:
        if classification.strip().upper().startswith("DETAILED"):
            return ResponsePlan(
                TurnIntent.ACKNOWLEDGE_THEN_RESPONSE,
                ResponseMode.DETAILED,
                acknowledge=True,
            )
        return ResponsePlan(TurnIntent.DIRECT_RESPONSE, ResponseMode.FAST)

    def plan(self, text: str) -> ResponsePlan:
        words = text.split()
        if not words:
            return ResponsePlan(TurnIntent.NO_RESPONSE, ResponseMode.FAST)
        if len(words) >= self._DETAILED_WORD_LIMIT or text.count("?") > 1:
            return ResponsePlan(
                TurnIntent.ACKNOWLEDGE_THEN_RESPONSE,
                ResponseMode.DETAILED,
                acknowledge=True,
            )
        if len(words) <= self._FAST_WORD_LIMIT:
            return ResponsePlan(TurnIntent.DIRECT_RESPONSE, ResponseMode.FAST)
        return ResponsePlan(TurnIntent.DIRECT_RESPONSE, ResponseMode.FAST)
