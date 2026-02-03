# AI Enterprise Data Analyst - OpenAI LLM Service
# Production-grade LLM integration with retry logic, streaming, and cost tracking

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Optional

from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError

from app.core.config import settings
from app.core.exceptions import OpenAIException, OpenAIRateLimitException
from app.core.logging import get_logger, LogContext

logger = get_logger(__name__)


# ============================================================================
# Types and Enums
# ============================================================================

class LLMModel(str, Enum):
    """Available LLM models."""
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


# Token costs per 1M tokens (approximate, update as needed)
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
        """Convert to OpenAI message format."""
        msg = {"role": self.role, "content": self.content}
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
    Production-grade OpenAI LLM service.
    
    Features:
    - Async operations with proper error handling
    - Automatic retry with exponential backoff
    - Token usage and cost tracking
    - Streaming support
    - Function/tool calling support
    - Embedding generation
    """
    
    _instance: Optional["LLMService"] = None
    
    def __new__(cls) -> "LLMService":
        """Singleton pattern for LLM service."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize OpenAI client."""
        if self._initialized:
            return
        
        self._client = AsyncOpenAI(
            api_key=settings.openai.api_key.get_secret_value(),
            organization=settings.openai.organization,
            timeout=settings.openai.timeout,
            max_retries=0  # We handle retries ourselves
        )
        
        self._default_model = settings.openai.default_model
        self._embedding_model = settings.openai.embedding_model
        self._total_usage = LLMUsage()
        self._initialized = True
        
        logger.info(
            "LLM Service initialized",
            default_model=self._default_model,
            embedding_model=self._embedding_model
        )
    
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
        
        Args:
            messages: List of chat messages
            model: Model to use (defaults to config)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            functions: Function definitions for function calling
            tools: Tool definitions for tool calling
            tool_choice: Tool choice strategy
            response_format: Response format (e.g., JSON mode)
            stop: Stop sequences
            **kwargs: Additional parameters
        
        Returns:
            LLMResponse with content and usage
        """
        model = model or self._default_model
        max_tokens = max_tokens or settings.openai.max_tokens_default
        
        context = LogContext(
            component="LLMService",
            operation="complete"
        )
        
        # Prepare request
        request_params = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if functions:
            request_params["functions"] = functions
        if tools:
            request_params["tools"] = tools
            if tool_choice:
                request_params["tool_choice"] = tool_choice
        if response_format:
            request_params["response_format"] = response_format
        if stop:
            request_params["stop"] = stop
        
        request_params.update(kwargs)
        
        # Execute with retry
        start_time = datetime.utcnow()
        
        for attempt in range(settings.openai.max_retries + 1):
            try:
                response = await self._client.chat.completions.create(**request_params)
                
                # Parse response
                choice = response.choices[0]
                usage = LLMUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    model=model
                )
                
                # Calculate cost
                try:
                    usage.calculate_cost(LLMModel(model))
                except ValueError:
                    pass  # Unknown model
                
                # Update cumulative usage
                self._total_usage.input_tokens += usage.input_tokens
                self._total_usage.output_tokens += usage.output_tokens
                self._total_usage.total_tokens += usage.total_tokens
                self._total_usage.estimated_cost_usd += usage.estimated_cost_usd
                
                latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                result = LLMResponse(
                    content=choice.message.content or "",
                    usage=usage,
                    model=model,
                    finish_reason=choice.finish_reason,
                    latency_ms=latency_ms
                )
                
                # Handle function/tool calls
                if choice.message.function_call:
                    result.function_call = {
                        "name": choice.message.function_call.name,
                        "arguments": choice.message.function_call.arguments
                    }
                
                if choice.message.tool_calls:
                    result.tool_calls = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in choice.message.tool_calls
                    ]
                
                logger.debug(
                    f"LLM completion successful",
                    context=context,
                    model=model,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    latency_ms=round(latency_ms, 2)
                )
                
                return result
                
            except RateLimitError as e:
                retry_after = getattr(e, "retry_after", 60)
                
                if attempt < settings.openai.max_retries:
                    logger.warning(
                        f"Rate limit hit, retrying in {retry_after}s",
                        context=context,
                        attempt=attempt + 1
                    )
                    await asyncio.sleep(retry_after)
                else:
                    raise OpenAIRateLimitException(retry_after=retry_after, cause=e)
                    
            except APITimeoutError as e:
                if attempt < settings.openai.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"API timeout, retrying in {wait_time}s",
                        context=context,
                        attempt=attempt + 1
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise OpenAIException(message="Request timed out", cause=e)
                    
            except APIError as e:
                logger.error(f"OpenAI API error: {e}", context=context)
                raise OpenAIException(message=str(e), cause=e)
    
    async def complete_stream(
        self,
        messages: list[Message],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming completion.
        
        Yields content chunks as they arrive.
        """
        model = model or self._default_model
        max_tokens = max_tokens or settings.openai.max_tokens_default
        
        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=[m.to_dict() for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except APIError as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise OpenAIException(message=str(e), cause=e)
    
    async def embed(
        self,
        texts: list[str],
        model: Optional[str] = None
    ) -> list[list[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
        
        Returns:
            List of embedding vectors
        """
        model = model or self._embedding_model
        
        context = LogContext(
            component="LLMService",
            operation="embed"
        )
        
        try:
            response = await self._client.embeddings.create(
                model=model,
                input=texts
            )
            
            embeddings = [data.embedding for data in response.data]
            
            logger.debug(
                f"Generated {len(embeddings)} embeddings",
                context=context,
                model=model,
                total_tokens=response.usage.total_tokens
            )
            
            return embeddings
            
        except APIError as e:
            logger.error(f"Embedding error: {e}", context=context)
            raise OpenAIException(message=str(e), cause=e)
    
    async def embed_single(self, text: str, model: Optional[str] = None) -> list[float]:
        """Generate embedding for single text."""
        embeddings = await self.embed([text], model)
        return embeddings[0]
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text (approximate).
        
        Uses a simple heuristic. For accurate counting, use tiktoken.
        """
        # Rough approximation: ~4 characters per token
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
