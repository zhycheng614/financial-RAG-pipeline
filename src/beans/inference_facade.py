from typing import List, Dict, Generator, Optional, AsyncGenerator
import time
from threading import Lock
from logging import getLogger

from constants import OPENAI_API_KEY
from data_classes.chat_completion_stream_response import (
    ChatCompletionStreamResponse,
    CompletionChoice,
    StreamDelta
)
from data_classes.inference_parameters import InferenceParameters


logger = getLogger(__name__)

class InferenceFacade():
    """Unified inference facade that uses the NexaAI SDK for all platforms."""
    
    def __init__(
        self, 
        model: str,
        base_url: str,
    ):
        """
        Initialize the facade with a model name and create openai client instance.
        
        Args:
            model: Model name.
            base_url: Base URL for the OpenAI-compatible service.
        """
        logger.debug(f"Initializing InferenceFacade with model: {model}, base_url: {base_url}")
        
        self.model = model
        self._lock = Lock()
        self._current_token_count = 0
        self._last_token_count = 0
        self._last_profiling: Optional[Dict] = None
    
        # Import openai components
        from openai import OpenAI, AsyncOpenAI
        
        # max_retries=10 (vs SDK default of 2) absorbs 429s on tier 2 keys
        # without crashing long sustained jobs (e.g. per-chunk LLM contextual
        # indexing, full benchmark sweeps).
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=base_url, max_retries=10)
        self.async_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=base_url, max_retries=10)
        
        logger.info(f"Successfully initialized InferenceFacade with model {model} at {base_url}")
    
    def _uses_max_completion_tokens(self) -> bool:
        """Newer OpenAI models (o1, o3, gpt-5+) require 'max_completion_tokens' instead of 'max_tokens'."""
        model_lower = self.model.lower()
        prefixes = ("o1", "o3", "o4", "gpt-5", "gpt-6", "gpt-7")
        return any(model_lower.startswith(p) for p in prefixes)
    
    def _stream_tokens(
        self, 
        messages: List[Dict[str, str]], 
        *, 
        inference_parameters: Optional[InferenceParameters] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Internal method to stream tokens."""
        with self._lock:
            # Handle system prompt override if provided
            if inference_parameters and inference_parameters.system_prompt:
                # Replace or add system prompt at the beginning
                messages = [msg for msg in messages if msg.get("role", "").lower() != "system"]
                messages = [{"role": "system", "content": inference_parameters.system_prompt}] + messages
            
            # Set parameters with defaults
            if inference_parameters:
                temperature = inference_parameters.temperature if inference_parameters.temperature is not None else 0.7
                max_tokens = inference_parameters.max_tokens if inference_parameters.max_tokens is not None else 2048
                top_p = inference_parameters.top_p if inference_parameters.top_p is not None else 0.95
                top_k = inference_parameters.top_k if inference_parameters.top_k is not None else 40
                repetition_penalty = inference_parameters.repetition_penalty if inference_parameters.repetition_penalty is not None else 1.0
                presence_penalty = inference_parameters.presence_penalty if inference_parameters.presence_penalty is not None else 0.0
                frequency_penalty = inference_parameters.frequency_penalty if inference_parameters.frequency_penalty is not None else 0.0
                stop_words = inference_parameters.stop_words
            else:
                # Use defaults
                temperature = kwargs.get('temperature', 0.7)
                max_tokens = kwargs.get('max_tokens', 2048)
                top_p = kwargs.get('top_p', 0.95)
                top_k = kwargs.get('top_k', 40)
                repetition_penalty = kwargs.get('repetition_penalty', 1.0)
                presence_penalty = kwargs.get('presence_penalty', 0.0)
                frequency_penalty = kwargs.get('frequency_penalty', 0.0)
                stop_words = kwargs.get('stop_words', None)
            
            # Build API parameters
            token_key = "max_completion_tokens" if self._uses_max_completion_tokens() else "max_tokens"
            api_params = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "temperature": temperature,
                token_key: max_tokens,
                "top_p": top_p,
            }
            
            # Add optional parameters
            if presence_penalty != 0.0:
                api_params["presence_penalty"] = presence_penalty
            if frequency_penalty != 0.0:
                api_params["frequency_penalty"] = frequency_penalty
            if stop_words:
                api_params["stop"] = stop_words
            if inference_parameters and inference_parameters.response_format:
                api_params["response_format"] = inference_parameters.response_format
            
            stream = self.client.chat.completions.create(**api_params)
            
            # Generate tokens
            try:
                # Reset current token count for this run
                self._current_token_count = 0
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        self._current_token_count += 1
                        yield token
                    
            except Exception as e:
                logger.error(f"Error during token generation: {e}")
                raise
            finally:
                # Persist token count for profiling retrieval
                self._last_token_count = self._current_token_count
    
    def stream_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        *,
        inference_parameters: Optional[InferenceParameters] = None,
        **kwargs
    ) -> Generator[ChatCompletionStreamResponse, None, None]:
        """Stream chat completion with TTFT recording logic."""
        start_time = time.time()
        first_token = True
        ttft_value: Optional[float] = None
        
        try:
            # Stream tokens and convert to response format
            token_gen = self._stream_tokens(
                messages,
                inference_parameters=inference_parameters,
                **kwargs
            )
            
            # Use a regular for loop to handle TTFT recording
            for i, token_str in enumerate(token_gen):
                # Create ChatCompletionStreamResponse
                response = ChatCompletionStreamResponse(
                    id=f"chatcmpl-{int(time.time())}",
                    model=self.model,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    choices=[
                        CompletionChoice(
                            index=0,
                            delta=StreamDelta(content=token_str if token_str else None),
                            finish_reason=None
                        )
                    ]
                )
                
                if first_token and token_str:
                    ttft = time.time() - start_time
                    ttft_value = ttft
                    logger.info(f"Time to first token (TTFT): {ttft:.3f}s")
                    first_token = False
                    
                yield response
            
            # Send final chunk with finish_reason
            yield ChatCompletionStreamResponse(
                id=f"chatcmpl-{int(time.time())}",
                model=self.model,
                object="chat.completion.chunk",
                created=int(time.time()),
                choices=[
                    CompletionChoice(
                        index=0,
                        delta=StreamDelta(content=None),
                        finish_reason="stop"
                    )
                ]
            )
            
        finally:
            # Log total inference time
            total_time = time.time() - start_time
            logger.info(f"Total inference time: {total_time:.3f}s")

            # Compute profiling metrics for this run
            try:
                effective_time = max(total_time - (ttft_value if ttft_value is not None else 0.0), 1e-6)
                tokens_generated = self._last_token_count
                tokens_per_second = tokens_generated / effective_time if effective_time > 0 else 0.0
                self._last_profiling = {
                    "time_to_first_token": float(ttft_value) if ttft_value is not None else None,
                    "tokens_per_second": float(tokens_per_second)
                }
            except Exception as e:
                logger.debug(f"Failed computing profiling metrics: {e}")
                self._last_profiling = None
    
    def create_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        *,
        inference_parameters: Optional[InferenceParameters] = None,
        **kwargs
    ) -> str:
        """Generate a complete chat completion response by collecting all tokens from streaming."""
        tokens = []
        
        for token_str in self._stream_tokens(
            messages,
            inference_parameters=inference_parameters,
            **kwargs
        ):
            if token_str:  # Only add non-empty tokens
                tokens.append(token_str)
        
        return "".join(tokens)
    
    def get_last_profiling(self) -> Optional[Dict]:
        """Get profiling metrics from the last inference run."""
        return self._last_profiling
    
    async def _stream_tokens_async(
        self, 
        messages: List[Dict[str, str]], 
        *, 
        inference_parameters: Optional[InferenceParameters] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Internal method to stream tokens asynchronously."""
        # Handle system prompt override if provided
        if inference_parameters and inference_parameters.system_prompt:
            # Replace or add system prompt at the beginning
            messages = [msg for msg in messages if msg.get("role", "").lower() != "system"]
            messages = [{"role": "system", "content": inference_parameters.system_prompt}] + messages
        
        # Set parameters with defaults
        if inference_parameters:
            temperature = inference_parameters.temperature if inference_parameters.temperature is not None else 0.7
            max_tokens = inference_parameters.max_tokens if inference_parameters.max_tokens is not None else 2048
            top_p = inference_parameters.top_p if inference_parameters.top_p is not None else 0.95
            top_k = inference_parameters.top_k if inference_parameters.top_k is not None else 40
            repetition_penalty = inference_parameters.repetition_penalty if inference_parameters.repetition_penalty is not None else 1.0
            presence_penalty = inference_parameters.presence_penalty if inference_parameters.presence_penalty is not None else 0.0
            frequency_penalty = inference_parameters.frequency_penalty if inference_parameters.frequency_penalty is not None else 0.0
            stop_words = inference_parameters.stop_words
        else:
            # Use defaults
            temperature = kwargs.get('temperature', 0.7)
            max_tokens = kwargs.get('max_tokens', 2048)
            top_p = kwargs.get('top_p', 0.95)
            top_k = kwargs.get('top_k', 40)
            repetition_penalty = kwargs.get('repetition_penalty', 1.0)
            presence_penalty = kwargs.get('presence_penalty', 0.0)
            frequency_penalty = kwargs.get('frequency_penalty', 0.0)
            stop_words = kwargs.get('stop_words', None)
        
        # Build API parameters
        token_key = "max_completion_tokens" if self._uses_max_completion_tokens() else "max_tokens"
        api_params = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            token_key: max_tokens,
            "top_p": top_p,
        }
        
        # Add optional parameters
        if presence_penalty != 0.0:
            api_params["presence_penalty"] = presence_penalty
        if frequency_penalty != 0.0:
            api_params["frequency_penalty"] = frequency_penalty
        if stop_words:
            api_params["stop"] = stop_words
        if inference_parameters and inference_parameters.response_format:
            api_params["response_format"] = inference_parameters.response_format
        
        stream = await self.async_client.chat.completions.create(**api_params)
        
        # Generate tokens
        try:
            # Reset current token count for this run
            self._current_token_count = 0
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    self._current_token_count += 1
                    yield token
                
        except Exception as e:
            logger.error(f"Error during async token generation: {e}")
            raise
        finally:
            # Persist token count for profiling retrieval
            self._last_token_count = self._current_token_count
    
    async def create_chat_completion_async(
        self, 
        messages: List[Dict[str, str]], 
        *,
        inference_parameters: Optional[InferenceParameters] = None,
        **kwargs
    ) -> str:
        """Generate a complete chat completion response asynchronously."""
        tokens = []
        
        async for token_str in self._stream_tokens_async(
            messages,
            inference_parameters=inference_parameters,
            **kwargs
        ):
            if token_str:  # Only add non-empty tokens
                tokens.append(token_str)
        
        return "".join(tokens)