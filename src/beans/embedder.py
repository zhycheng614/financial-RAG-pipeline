from typing import List
from data_classes.embedding_generator_config import EmbedderConfig
from logging import getLogger
from utils.math import l2_normalize
from constants import OPENAI_API_KEY, DEFAULT_EMBEDDING_DIM
import numpy as np
from openai import OpenAI, AsyncOpenAI
from functools import wraps
from typing import Union

logger = getLogger(__name__)

# max_retries=10 is needed for sustained large-batch embedding work on tier 2
# accounts; the SDK default of 2 retries gives up too quickly when the 1 M TPM
# embeddings limit is hit, even though the server returns retry-after windows
# under one second.
client = OpenAI(
    api_key=OPENAI_API_KEY,
    max_retries=10,
)

async_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    max_retries=10,
)

def validate_non_empty_text(func):
    """
    Decorator to validate that text inputs are not empty or whitespace-only.
    Works with both single text strings and lists of text strings.
    """
    @wraps(func)
    def wrapper(self, texts: Union[str, List[str]], *args, **kwargs):
        # Handle single text string
        if isinstance(texts, str):
            if not texts or not texts.strip():
                raise ValueError("Cannot generate embedding for empty or whitespace-only text")
        # Handle list of text strings
        elif isinstance(texts, list):
            if len(texts) == 0:
                raise ValueError("Cannot generate embeddings for empty text list")
            
            empty_indices = []
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    empty_indices.append(i)
            
            if empty_indices:
                raise ValueError(f"Cannot generate embeddings for empty or whitespace-only texts at indices: {empty_indices}")
        
        return func(self, texts, *args, **kwargs)
    return wrapper


class Embedder():
    """
    Embedder that uses OpenAI as backend for embedding generation
    """
    def __init__(self, config: EmbedderConfig):
        self.config = config

    @validate_non_empty_text
    def generate_embedding(self, text: str) -> np.ndarray:
        return self.generate_embeddings([text])[0]
        
    @validate_non_empty_text
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        
        for i in range(0, len(texts), self.config.max_batch_size):
            batch = texts[i:i + self.config.max_batch_size]
            batch_embeddings = self._embed_text_lists(batch)
            all_embeddings.append(batch_embeddings)
            logger.debug(f"Completed batch {i}:{i + batch_embeddings.shape[0]}")
            
        final_embeddings = np.vstack(all_embeddings)
        logger.debug(f"Final embeddings shape: {final_embeddings.shape}")
        
        # Truncate the embeddings and normalize
        truncated_embeddings = final_embeddings
        return l2_normalize(truncated_embeddings)
    
    def _embed_text_lists(self, text_list: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of text using OpenAI.
        On Windows ARM64, uses direct text input. Otherwise, tokenizes on our side.
        """
        logger.debug(f"embeddings text_list: { text_list }")
        try:
            completion = client.embeddings.create(
                # model="text-embedding-v4",
                model=self.config.model_name,
                # model="text-embedding-3-large",
                # model="text-embedding-ada-002",
                input=text_list,
                dimensions=self.config.embedding_dim,
                encoding_format="float"
            )
            embeddings = []
            for embedding_temp_data in completion.data:
                embeddings.append(embedding_temp_data.embedding)
            embeddings = np.array(embeddings, dtype=np.float32)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {e}")
            raise e
    
    async def _embed_text_lists_async(self, text_list: List[str]) -> np.ndarray:
        """
        Asynchronously generate embeddings for a batch of text using OpenAI.
        """
        logger.debug(f"embeddings text_list (async): { text_list }")
        try:
            completion = await async_client.embeddings.create(
                model=self.config.model_name,
                input=text_list,
                dimensions=self.config.embedding_dim,
                encoding_format="float"
            )
            embeddings = []
            for embedding_temp_data in completion.data:
                embeddings.append(embedding_temp_data.embedding)
            embeddings = np.array(embeddings, dtype=np.float32)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI (async): {e}")
            raise e
    
    @validate_non_empty_text
    async def generate_embedding_async(self, text: str) -> np.ndarray:
        """Asynchronously generate embedding for a single text."""
        embeddings = await self.generate_embeddings_async([text])
        return embeddings[0]
    
    @validate_non_empty_text
    async def generate_embeddings_async(self, texts: List[str]) -> np.ndarray:
        """Asynchronously generate embeddings for multiple texts."""
        all_embeddings = []
        
        for i in range(0, len(texts), self.config.max_batch_size):
            batch = texts[i:i + self.config.max_batch_size]
            batch_embeddings = await self._embed_text_lists_async(batch)
            all_embeddings.append(batch_embeddings)
            logger.debug(f"Completed batch {i}:{i + batch_embeddings.shape[0]} (async)")
            
        final_embeddings = np.vstack(all_embeddings)
        logger.debug(f"Final embeddings shape (async): {final_embeddings.shape}")
        
        # Truncate the embeddings and normalize
        truncated_embeddings = final_embeddings
        return l2_normalize(truncated_embeddings)