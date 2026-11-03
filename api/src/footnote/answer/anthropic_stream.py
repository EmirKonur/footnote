"""The production streamer: Claude through the Anthropic SDK."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

from anthropic import AsyncAnthropic
from anthropic.lib.streaming import BetaAsyncMessageStream
from anthropic.types.beta import (
    BetaRawContentBlockDeltaEvent,
    BetaRawContentBlockStartEvent,
    BetaRawMessageDeltaEvent,
    BetaRawMessageStartEvent,
)

from footnote.answer.generate import GenerationRequest, StreamEvent
from footnote.answer.prompt import build_messages, system_blocks
from footnote.config import Effort
from footnote.schemas import GenerationMode

# The "default" fallback mode lets the API pick the substitute model by refusal
# category, so there is no model list here to keep current. It needs this exact beta;
# the older array form of `fallbacks` uses server-side-fallback-2026-06-01 instead.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def has_credentials(client: AsyncAnthropic) -> bool:
    """True when the SDK resolved some credential (key, token, profile, or federation)
    using its own precedence rules."""
    return bool(client.api_key or client.auth_token or client.credentials is not None)


class AnthropicStreamer:
    def __init__(
        self, client: AsyncAnthropic, *, model: str, effort: Effort, max_tokens: int
    ) -> None:
        self._client = client
        self._model = model
        self._effort: Effort = effort
        self._max_tokens = max_tokens

    @property
    def mode(self) -> GenerationMode:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    @asynccontextmanager
    async def open(self, request: GenerationRequest) -> AsyncGenerator[AsyncIterator[StreamEvent]]:
        # Deliberately absent: temperature/top_p/top_k (rejected on API version
        # 2026-10-21), output_config.format (incompatible with citations), and any
        # thinking budget (Opus 5.5 only takes adaptive thinking; effort sets depth).
        async with self._client.beta.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_blocks(),
            messages=build_messages(request.question, request.passages, request.history),
            thinking={"type": "adaptive"},
            output_config={"effort": self._effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            yield raw_events(stream)


async def raw_events[T](stream: BetaAsyncMessageStream[T]) -> AsyncIterator[StreamEvent]:
    """Keep the four raw event types the answer builder reads; the SDK interleaves its
    own convenience events (text, citation, thinking) that duplicate them."""
    async for event in stream:
        if isinstance(
            event,
            BetaRawMessageStartEvent
            | BetaRawMessageDeltaEvent
            | BetaRawContentBlockStartEvent
            | BetaRawContentBlockDeltaEvent,
        ):
            yield event
