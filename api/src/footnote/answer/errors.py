"""Turn Anthropic SDK exceptions into messages a user can act on."""

from __future__ import annotations

from dataclasses import dataclass

import anthropic


@dataclass(frozen=True, slots=True)
class GenerationFailure:
    code: str
    message: str
    retryable: bool


def describe_failure(exc: anthropic.AnthropicError, model: str) -> GenerationFailure:
    """Most specific class first: several of these subclass each other (a timeout is a
    connection error, and every HTTP error is an APIStatusError)."""
    if isinstance(exc, anthropic.AuthenticationError):
        return GenerationFailure(
            "auth_failed",
            "The Anthropic API rejected the credentials. Check ANTHROPIC_API_KEY.",
            False,
        )
    if isinstance(exc, anthropic.PermissionDeniedError):
        return GenerationFailure("permission_denied", f"This key may not use {model}.", False)
    if isinstance(exc, anthropic.NotFoundError):
        return GenerationFailure(
            "model_not_found", f"Model {model} is not available. Check FOOTNOTE_MODEL.", False
        )
    if isinstance(exc, anthropic.RateLimitError):
        return GenerationFailure(
            "rate_limited", "Rate limited by the Anthropic API. Try again shortly.", True
        )
    if isinstance(exc, anthropic.OverloadedError):
        return GenerationFailure(
            "overloaded", "The Anthropic API is overloaded. Try again shortly.", True
        )
    if isinstance(exc, anthropic.RequestTooLargeError):
        return GenerationFailure(
            "request_too_large", "The question and passages are too large for one request.", False
        )
    if isinstance(exc, anthropic.BadRequestError):
        return GenerationFailure(
            "bad_request", f"The API rejected the request: {exc.message}", False
        )
    if isinstance(exc, anthropic.APIStatusError):
        retryable = exc.status_code >= 500
        return GenerationFailure(
            "upstream_error", f"The Anthropic API returned HTTP {exc.status_code}.", retryable
        )
    if isinstance(exc, anthropic.APITimeoutError):
        return GenerationFailure("timeout", "The Anthropic API did not respond in time.", True)
    if isinstance(exc, anthropic.APIConnectionError):
        return GenerationFailure("connection_failed", "Could not reach the Anthropic API.", True)
    if isinstance(exc, anthropic.CredentialsError):
        return GenerationFailure("auth_missing", f"No usable Anthropic credentials: {exc}", False)
    return GenerationFailure("unexpected", f"Generation failed: {exc}", False)
