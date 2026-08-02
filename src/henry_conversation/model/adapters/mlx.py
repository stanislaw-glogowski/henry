import copy
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ...config import LanguageModelProfile, LanguageModelsProfile
from ...domain import LanguageModelChunk, LanguageModelRequest, LanguageModelRole
from ..ports import LanguageModel


@dataclass(slots=True)
class _LoadedModel:
    model: Any
    tokenizer: Any
    prompt_caches: dict[str, tuple[tuple[int, ...], Any]]


class MLXLanguageModel(LanguageModel):
    def __init__(self, profiles: LanguageModelsProfile) -> None:
        super().__init__()
        self._profiles = profiles
        self._models: dict[str, _LoadedModel] = {}
        self._load: Any = None
        self._stream_generate: Any = None
        self._make_sampler: Any = None
        self._make_prompt_cache: Any = None
        self._generate_step: Any = None
        self._array: Any = None

    def open(self) -> None:
        if self._load is not None:
            raise RuntimeError("MLX language model adapter is already open")
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        self._load = load
        self._stream_generate = stream_generate
        self._make_sampler = make_sampler
        self._make_prompt_cache = make_prompt_cache
        self._generate_step = generate_step
        self._array = mx.array
        self._logger.debug("Adapter OPENED")

    def close(self) -> None:
        self._models.clear()
        self._load = None
        self._stream_generate = None
        self._make_sampler = None
        self._make_prompt_cache = None
        self._generate_step = None
        self._array = None
        self._logger.debug("Adapter CLOSED")

    def prepare(self, role: LanguageModelRole) -> None:
        self._model(role)

    def generate(self, request: LanguageModelRequest) -> Iterator[LanguageModelChunk]:
        loaded = self._model(request.role)
        profile = self._profile(request.role)
        prompt = loaded.tokenizer.apply_chat_template(
            [
                {"role": message.role.value, "content": message.content}
                for message in request.messages
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=profile.thinking,
        )
        if self._stream_generate is None or self._make_sampler is None:
            raise RuntimeError("MLX language model adapter is not open")
        sampler = self._make_sampler(
            temp=profile.temperature,
            top_p=profile.top_p,
            top_k=profile.top_k,
        )
        prompt_value, prompt_cache = self._cached_prompt(
            loaded,
            request,
            prompt,
            profile.thinking,
        )
        for response in self._stream_generate(
            loaded.model,
            loaded.tokenizer,
            prompt=prompt_value,
            max_tokens=profile.max_tokens,
            sampler=sampler,
            prompt_cache=prompt_cache,
        ):
            if response.text:
                yield LanguageModelChunk(response.text)

    def _model(self, role: LanguageModelRole) -> _LoadedModel:
        profile = self._profile(role)
        model_id = profile.model_for("mlx")
        if loaded := self._models.get(model_id):
            return loaded
        if self._load is None:
            raise RuntimeError("MLX language model adapter is not open")
        model, tokenizer = self._load(model_id)
        loaded = _LoadedModel(model, tokenizer, {})
        self._models[model_id] = loaded
        self._logger.debug("Model LOADED: role='{}', model='{}'", role, model_id)
        return loaded

    def _cached_prompt(
        self,
        loaded: _LoadedModel,
        request: LanguageModelRequest,
        prompt: str,
        thinking: bool,
    ) -> tuple[str | list[int], Any]:
        if (
            len(request.messages) < 2
            or request.messages[0].role.value != "system"
            or self._make_prompt_cache is None
            or self._generate_step is None
            or self._array is None
        ):
            return prompt, None

        system_content = request.messages[0].content
        cache_key = f"{thinking}:{system_content}"
        if cached := loaded.prompt_caches.get(cache_key):
            prefix_tokens, prompt_cache = cached
        else:
            try:
                prefix = loaded.tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_content}],
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=thinking,
                )
            except Exception as error:
                self._logger.debug(
                    "Prompt prefix cache unavailable: {}", type(error).__name__
                )
                return prompt, None
            prefix_tokens = tuple(loaded.tokenizer.encode(prefix))
            prompt_cache = self._make_prompt_cache(loaded.model)
            for _ in self._generate_step(
                self._array(prefix_tokens),
                loaded.model,
                max_tokens=0,
                prompt_cache=prompt_cache,
            ):
                pass
            loaded.prompt_caches[cache_key] = (prefix_tokens, prompt_cache)

        prompt_tokens = loaded.tokenizer.encode(prompt)
        prefix_length = len(prefix_tokens)
        if tuple(prompt_tokens[:prefix_length]) != prefix_tokens:
            return prompt, None
        return prompt_tokens[prefix_length:], copy.deepcopy(prompt_cache)

    def _profile(self, role: LanguageModelRole) -> LanguageModelProfile:
        profile = getattr(self._profiles, role.value)
        if profile is None:
            raise RuntimeError(f"Language model role is not configured: {role!r}")
        return profile
