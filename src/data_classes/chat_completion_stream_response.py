from dataclasses import dataclass
from typing import List, Literal, Optional, Dict


@dataclass
class StreamDelta:
    """Delta content in streaming response."""
    content: Optional[str] = None


@dataclass
class CompletionChoice:
    """Completion choice in streaming response."""
    index: int
    delta: StreamDelta
    logprobs: Optional[Dict[str, str]] = None
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletionStreamResponse:
    """Chat completion streaming response format."""
    id: str
    model: str
    object: Literal["chat.completion.chunk"]
    created: int
    choices: List[CompletionChoice]