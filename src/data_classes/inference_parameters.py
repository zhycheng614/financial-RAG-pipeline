from dataclasses import dataclass
from typing import Optional, List


@dataclass
class InferenceParameters:
    """
    Parameters for controlling inference behavior.
    All fields are optional - only provided fields will override defaults.
    Based on nexaai's SamplerConfig and GenerationConfig.
    Excludes image_paths and audio_paths which are handled separately for VLM.
    """
    # Sampler parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    grammar_path: Optional[str] = None
    grammar_string: Optional[str] = None
    
    # Generation parameters
    max_tokens: Optional[int] = None
    stop_words: Optional[List[str]] = None
    
    # Response format (e.g., {"type": "json_object"} for JSON mode)
    response_format: Optional[dict] = None
    
    # System prompt override (has highest priority, disables KV cache)
    system_prompt: Optional[str] = None