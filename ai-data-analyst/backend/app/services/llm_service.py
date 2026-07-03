# AI Enterprise Data Analyst - LLM Service
# Provider-agnostic LLM integration via LiteLLM
# Supports 100+ providers: OpenAI, Anthropic, Google, Mistral, Ollama, etc.

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Optional

# ── Lazy-loaded LLM libraries (imported on first use, not at startup) ─────
# LiteLLM is 14.6 MB and eagerly loads 100+ provider SDKs.
# Importing at module level blocks server startup for minutes.
_litellm = None        # lazily loaded litellm module
_acompletion = None    # litellm.acompletion
_aembedding = None     # litellm.aembedding
_HAS_LITELLM: bool | None = None  # None = not yet checked
_HAS_OPENAI: bool | None = None
_AsyncOpenAI = None    # openai.AsyncOpenAI


def _ensure_litellm() -> bool:
    """Lazy-load litellm on first call. Returns True if available."""
    global _litellm, _acompletion, _aembedding, _HAS_LITELLM
    if _HAS_LITELLM is not None:
        return _HAS_LITELLM
    try:
        import litellm as _ll
        from litellm import acompletion, aembedding
        _ll.drop_params = True
        _ll.set_verbose = False
        _litellm = _ll
        _acompletion = acompletion
        _aembedding = aembedding
        _HAS_LITELLM = True
    except ImportError:
        _HAS_LITELLM = False
    return _HAS_LITELLM


def _ensure_openai() -> bool:
    """Lazy-load openai SDK on first call. Returns True if available."""
    global _AsyncOpenAI, _HAS_OPENAI
    if _HAS_OPENAI is not None:
        return _HAS_OPENAI
    try:
        from openai import AsyncOpenAI
        _AsyncOpenAI = AsyncOpenAI
        _HAS_OPENAI = True
    except ImportError:
        _HAS_OPENAI = False
    return _HAS_OPENAI

from app.core.config import get_settings
from app.core.exceptions import OpenAIException, OpenAIRateLimitException
from app.core.logging import get_logger, LogContext

logger = get_logger(__name__)


# ============================================================================
# Types and Enums
# ============================================================================

class LLMModel(str, Enum):
    """Well-known LLM models (non-exhaustive, any string works for LiteLLM)."""
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_35_TURBO = "gpt-3.5-turbo"
    O1 = "o1"
    O1_MINI = "o1-mini"
    O1_PREVIEW = "o1-preview"


class EmbeddingModel(str, Enum):
    """Available embedding models."""
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_ADA = "text-embedding-ada-002"


# Token costs per 1M tokens (approximate, LiteLLM handles this too)
MODEL_COSTS = {
    LLMModel.GPT_4O: {"input": 5.0, "output": 15.0},
    LLMModel.GPT_4O_MINI: {"input": 0.15, "output": 0.6},
    LLMModel.GPT_4_TURBO: {"input": 10.0, "output": 30.0},
    LLMModel.GPT_35_TURBO: {"input": 0.5, "output": 1.5},
    LLMModel.O1: {"input": 15.0, "output": 60.0},
    LLMModel.O1_MINI: {"input": 3.0, "output": 12.0},
}


@dataclass
class LLMUsage:
    """Token usage tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    estimated_cost_usd: float = 0.0

    def calculate_cost(self, model: LLMModel) -> float:
        """Calculate estimated cost in USD."""
        if model in MODEL_COSTS:
            costs = MODEL_COSTS[model]
            input_cost = (self.input_tokens / 1_000_000) * costs["input"]
            output_cost = (self.output_tokens / 1_000_000) * costs["output"]
            self.estimated_cost_usd = input_cost + output_cost
        return self.estimated_cost_usd


@dataclass
class LLMResponse:
    """Response from LLM completion."""

    content: str
    usage: LLMUsage
    model: str
    finish_reason: str
    function_call: Optional[dict[str, Any]] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    raw_response: Optional[dict[str, Any]] = None
    latency_ms: float = 0.0


@dataclass
class Message:
    """Chat message for LLM."""

    role: str  # system, user, assistant, function, tool
    content: str
    name: Optional[str] = None
    function_call: Optional[dict[str, Any]] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible message format."""
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.function_call:
            msg["function_call"] = self.function_call
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


# ============================================================================
# LLM Service Implementation
# ============================================================================

class LLMService:
    """
    Provider-agnostic LLM service via LiteLLM.

    Features:
    - Supports 100+ LLM providers (OpenAI, Anthropic, Google, Ollama, etc.)
    - Async operations with proper error handling
    - Automatic retry with exponential backoff
    - Token usage and cost tracking
    - Streaming support
    - Function/tool calling support
    - Embedding generation
    - Falls back to direct OpenAI SDK if LiteLLM is not installed
    """

    _instance: Optional["LLMService"] = None

    def __new__(cls) -> "LLMService":
        """Singleton pattern for LLM service."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize LLM service (lightweight — no heavy imports)."""
        if self._initialized:
            return

        settings = get_settings()
        self._default_model = settings.llm_model or settings.openai.default_model
        self._embedding_model = settings.openai.embedding_model
        self._max_retries = settings.openai.max_retries
        self._timeout = settings.openai.timeout
        self._max_tokens_default = settings.openai.max_tokens_default
        self._total_usage = LLMUsage()
        self._use_litellm = None  # resolved lazily
        self._client = None
        self._provider_loaded = False

        # Resolve API key: prefer openai.api_key if set, else check env
        api_key_secret = settings.openai.api_key
        self._api_key = api_key_secret.get_secret_value() if api_key_secret else None

        logger.info(
            "LLM Service initialized (provider loaded on first call)",
            default_model=self._default_model,
        )
        self._initialized = True

    def _load_provider(self) -> None:
        """Lazy-load the LLM provider on first actual LLM call."""
        if self._provider_loaded:
            return
        self._provider_loaded = True

        if _ensure_litellm():
            self._use_litellm = True
            if self._api_key:
                import os
                os.environ.setdefault("OPENAI_API_KEY", self._api_key)
            logger.info("LLM provider loaded: LiteLLM", model=self._default_model)
        elif _ensure_openai() and self._api_key:
            self._use_litellm = False
            self._client = _AsyncOpenAI(
                api_key=self._api_key,
                organization=get_settings().openai.organization,
                timeout=self._timeout,
                max_retries=0,
            )
            logger.info("LLM provider loaded: OpenAI direct", model=self._default_model)
        else:
            self._use_litellm = False
            logger.warning(
                "LLM provider: NONE. "
                "Set OPENAI_API_KEY or install litellm + configure LLM_MODEL."
            )

    @property
    def is_available(self) -> bool:
        """Check if LLM is configured and available."""
        self._load_provider()
        if self._use_litellm:
            return True  # LiteLLM can use Ollama etc. without API keys
        return self._api_key is not None and (_HAS_OPENAI or False)

    @property
    def total_usage(self) -> LLMUsage:
        """Get cumulative token usage."""
        return self._total_usage

    async def complete(
        self,
        messages: list[Message],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        functions: Optional[list[dict[str, Any]]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str | dict[str, Any]] = None,
        response_format: Optional[dict[str, Any]] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """
        Generate completion with retry logic.

        Works with any LiteLLM-supported model string:
        - "gpt-4o" (OpenAI)
        - "claude-sonnet-4-20250514" (Anthropic)
        - "gemini/gemini-pro" (Google)
        - "ollama/llama3" (local Ollama)
        - etc.
        """
        model = model or self._default_model
        max_tokens = max_tokens or self._max_tokens_default

        if not self.is_available:
            raise OpenAIException(
                message="LLM is not configured. Set LLM_MODEL and the appropriate API key, "
                        "or install litellm with 'pip install litellm'."
            )

        context = LogContext(component="LLMService", operation="complete")
        msg_dicts = [m.to_dict() for m in messages]
        start_time = datetime.utcnow()

        for attempt in range(self._max_retries + 1):
            try:
                if self._use_litellm:
                    response = await self._litellm_complete(
                        model=model,
                        messages=msg_dicts,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        functions=functions,
                        tools=tools,
                        tool_choice=tool_choice,
                        response_format=response_format,
                        stop=stop,
                        **kwargs,
                    )
                else:
                    response = await self._openai_complete(
                        model=model,
                        messages=msg_dicts,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        functions=functions,
                        tools=tools,
                        tool_choice=tool_choice,
                        response_format=response_format,
                        stop=stop,
                        **kwargs,
                    )

                latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                response.latency_ms = latency_ms

                # Update cumulative usage
                self._total_usage.input_tokens += response.usage.input_tokens
                self._total_usage.output_tokens += response.usage.output_tokens
                self._total_usage.total_tokens += response.usage.total_tokens
                self._total_usage.estimated_cost_usd += response.usage.estimated_cost_usd

                logger.debug(
                    "LLM completion successful",
                    context=context,
                    model=model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_ms=round(latency_ms, 2),
                )
                return response

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "rate" in err_str and "limit" in err_str
                is_timeout = "timeout" in err_str

                if is_rate_limit and attempt < self._max_retries:
                    wait = 60
                    logger.warning(f"Rate limit hit, retrying in {wait}s", context=context, attempt=attempt + 1)
                    await asyncio.sleep(wait)
                elif is_timeout and attempt < self._max_retries:
                    wait = 2 ** attempt
                    logger.warning(f"Timeout, retrying in {wait}s", context=context, attempt=attempt + 1)
                    await asyncio.sleep(wait)
                elif is_rate_limit:
                    raise OpenAIRateLimitException(retry_after=60, cause=e)
                else:
                    logger.error(f"LLM API error: {e}", context=context)
                    raise OpenAIException(message=str(e), cause=e)

        # Should not reach here, but just in case
        raise OpenAIException(message="Max retries exceeded")

    async def _litellm_complete(self, *, model: str, messages: list[dict], **kwargs: Any) -> LLMResponse:
        """Complete via LiteLLM (supports 100+ providers)."""
        # Build params, removing None values
        params: dict[str, Any] = {"model": model, "messages": messages}
        for k, v in kwargs.items():
            if v is not None:
                params[k] = v

        response = await _acompletion(**params)
        choice = response.choices[0]

        usage = LLMUsage(
            input_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(response.usage, "total_tokens", 0) or 0,
            model=model,
        )
        try:
            usage.calculate_cost(LLMModel(model))
        except ValueError:
            pass

        result = LLMResponse(
            content=choice.message.content or "",
            usage=usage,
            model=model,
            finish_reason=choice.finish_reason or "stop",
        )

        if hasattr(choice.message, "function_call") and choice.message.function_call:
            result.function_call = {
                "name": choice.message.function_call.name,
                "arguments": choice.message.function_call.arguments,
            }
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            result.tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.message.tool_calls
            ]

        return result

    async def _openai_complete(self, *, model: str, messages: list[dict], **kwargs: Any) -> LLMResponse:
        """Complete via direct OpenAI SDK (fallback)."""
        params: dict[str, Any] = {"model": model, "messages": messages}
        for k in ("temperature", "max_tokens", "stop"):
            if kwargs.get(k) is not None:
                params[k] = kwargs[k]
        if kwargs.get("functions"):
            params["functions"] = kwargs["functions"]
        if kwargs.get("tools"):
            params["tools"] = kwargs["tools"]
            if kwargs.get("tool_choice"):
                params["tool_choice"] = kwargs["tool_choice"]
        if kwargs.get("response_format"):
            params["response_format"] = kwargs["response_format"]

        response = await self._client.chat.completions.create(**params)
        choice = response.choices[0]

        usage = LLMUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            model=model,
        )
        try:
            usage.calculate_cost(LLMModel(model))
        except ValueError:
            pass

        result = LLMResponse(
            content=choice.message.content or "",
            usage=usage,
            model=model,
            finish_reason=choice.finish_reason,
        )

        if choice.message.function_call:
            result.function_call = {
                "name": choice.message.function_call.name,
                "arguments": choice.message.function_call.arguments,
            }
        if choice.message.tool_calls:
            result.tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.message.tool_calls
            ]

        return result

    async def complete_stream(
        self,
        messages: list[Message],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Generate streaming completion. Yields content chunks."""
        model = model or self._default_model
        max_tokens = max_tokens or self._max_tokens_default

        if not self.is_available:
            yield "LLM is not configured. Set LLM_MODEL and the appropriate API key."
            return

        try:
            if self._use_litellm:
                response = await _acompletion(
                    model=model,
                    messages=[m.to_dict() for m in messages],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    **kwargs,
                )
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            elif self._client:
                stream = await self._client.chat.completions.create(
                    model=model,
                    messages=[m.to_dict() for m in messages],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    **kwargs,
                )
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise OpenAIException(message=str(e), cause=e)

    async def embed(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[list[float]]:
        """Generate embeddings for texts."""
        model = model or self._embedding_model
        context = LogContext(component="LLMService", operation="embed")

        try:
            if self._use_litellm:
                response = await _aembedding(model=model, input=texts)
                embeddings = [data["embedding"] for data in response.data]
            elif self._client:
                response = await self._client.embeddings.create(model=model, input=texts)
                embeddings = [data.embedding for data in response.data]
            else:
                raise OpenAIException(message="No LLM provider configured for embeddings")

            logger.debug(f"Generated {len(embeddings)} embeddings", context=context, model=model)
            return embeddings

        except OpenAIException:
            raise
        except Exception as e:
            logger.error(f"Embedding error: {e}", context=context)
            raise OpenAIException(message=str(e), cause=e)

    async def embed_single(self, text: str, model: Optional[str] = None) -> list[float]:
        """Generate embedding for single text."""
        embeddings = await self.embed([text], model)
        return embeddings[0]

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text (approximate)."""
        return len(text) // 4


# Factory function
def get_llm_service() -> LLMService:
    """Get LLM service instance."""
    return LLMService()


# Convenience functions
async def complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    **kwargs: Any
) -> str:
    """Simple completion helper."""
    service = get_llm_service()
    messages = []
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))
    messages.append(Message(role="user", content=prompt))
    response = await service.complete(messages, **kwargs)
    return response.content


async def embed(texts: list[str]) -> list[list[float]]:
    """Simple embedding helper."""
    service = get_llm_service()
    return await service.embed(texts)
