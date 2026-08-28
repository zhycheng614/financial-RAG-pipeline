from logging import getLogger
from typing import List, Optional
import json
import re
from beans.inference_facade import InferenceFacade
from data_classes.inference_parameters import InferenceParameters
from data_classes.rewrite_service_output import RewriteServiceOutput
from data_classes.ticker_extraction_output import TickerExtractionOutput, TickerYearInfo
from enums.message_role import MessageRole
from prompts import (
    REWRITING_PROMPT, 
    REWRITING_USER_QUERY_PROMPT,
    TICKER_EXTRACTION_SYSTEM_PROMPT,
    TICKER_EXTRACTION_USER_PROMPT
)

logger = getLogger(__name__)

class RewritingService:
    def __init__(self, inference_facade: InferenceFacade):
        self.inference_facade = inference_facade
    
    
    def rewrite(self, query: str) -> RewriteServiceOutput:
        
        updated_messages = [
            {
                "role": MessageRole.SYSTEM.value,
                "content": REWRITING_PROMPT
            },
            {
                "role": MessageRole.USER.value,
                "content": REWRITING_USER_QUERY_PROMPT.format(query=query),
            },
        ]

        inference_parameters = InferenceParameters(
            response_format={"type": "json_object"}
        )
        
        json_str = self.inference_facade.create_chat_completion(
            messages=updated_messages,
            inference_parameters=inference_parameters
        )
        
        # Parse JSON and create dataclass instance
        try:
            parsed_json = json.loads(json_str)
            rewrite_service_output = RewriteServiceOutput(**parsed_json)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            # Handle invalid JSON or missing fields
            logger.error(f"Error parsing rewrite service output: {e}")
            logger.error(f"JSON string: {json_str}")
            
            # Fallback: use original query
            logger.warning("Falling back to original query without rewriting")
            rewrite_service_output = RewriteServiceOutput(
                clarified_query=query,
                keywords=self._extract_keywords_from_query(query)
            )
        
        logger.debug(f"Rewrite service output: {rewrite_service_output}")
        return rewrite_service_output
    
    async def rewrite_async(self, query: str) -> RewriteServiceOutput:
        """Asynchronous version of rewrite method."""
        
        updated_messages = [
            {
                "role": MessageRole.SYSTEM.value,
                "content": REWRITING_PROMPT
            },
            {
                "role": MessageRole.USER.value,
                "content": REWRITING_USER_QUERY_PROMPT.format(query=query),
            },
        ]

        inference_parameters = InferenceParameters(
            response_format={"type": "json_object"}
        )
        
        json_str = await self.inference_facade.create_chat_completion_async(
            messages=updated_messages,
            inference_parameters=inference_parameters
        )
        
        # Parse JSON and create dataclass instance
        try:
            parsed_json = json.loads(json_str)
            rewrite_service_output = RewriteServiceOutput(**parsed_json)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            # Handle invalid JSON or missing fields
            logger.error(f"Error parsing rewrite service output: {e}")
            logger.error(f"JSON string: {json_str}")
            
            # Fallback: use original query
            logger.warning("Falling back to original query without rewriting")
            rewrite_service_output = RewriteServiceOutput(
                clarified_query=query,
                keywords=self._extract_keywords_from_query(query)
            )
        
        logger.debug(f"Rewrite service output (async): {rewrite_service_output}")
        return rewrite_service_output

    
    def _extract_keywords_from_query(self, query: str) -> List[str]:
        """
        Extract keywords from query by splitting on spaces and removing special characters.
        
        Args:
            query: The raw query string
            
        Returns:
            List of cleaned keywords
        """
        # Split by spaces
        words = query.split()
        
        # Remove special characters from each word and filter out empty strings
        keywords = []
        for word in words:
            # Remove all non-alphanumeric characters (keep letters and numbers)
            cleaned_word = re.sub(r'[^a-zA-Z0-9]', '', word)
            if cleaned_word:  # Only add non-empty words
                keywords.append(cleaned_word)
        
        # If no keywords after cleaning, fall back to original query
        if not keywords:
            keywords = [query]
        
        return keywords
    
    def extract_ticker_info(self, query: str, default_year: int, max_retries: int = 3) -> TickerExtractionOutput:
        """
        Extract ticker symbols and years from a query for filename-based document retrieval.
        
        Args:
            query: The user query about financial reports
            default_year: Default year to use if no year is specified in the query
            max_retries: Maximum number of retry attempts if JSON parsing fails
            
        Returns:
            TickerExtractionOutput containing list of ticker-year pairs
        """
        # Format system prompt with default year
        system_prompt = TICKER_EXTRACTION_SYSTEM_PROMPT.format(default_year=default_year)
        
        updated_messages = [
            {
                "role": MessageRole.SYSTEM.value,
                "content": system_prompt
            },
            {
                "role": MessageRole.USER.value,
                "content": TICKER_EXTRACTION_USER_PROMPT.format(query=query),
            },
        ]

        inference_parameters = InferenceParameters(
            response_format={"type": "json_object"}
        )
        
        last_error = None
        last_json_str = None
        
        for attempt in range(max_retries):
            json_str = self.inference_facade.create_chat_completion(
                messages=updated_messages,
                inference_parameters=inference_parameters
            )
            last_json_str = json_str
            
            # Try to parse JSON response
            try:
                result = self._parse_ticker_extraction_response(json_str)
                if result is not None:
                    logger.debug(f"Ticker extraction output: {result}")
                    return result
            except (json.JSONDecodeError, TypeError, KeyError, ValueError) as e:
                last_error = e
                # Always log problematic JSON immediately (not just in verbose mode)
                logger.error(f"[Attempt {attempt + 1}/{max_retries}] Failed to parse ticker extraction JSON: {e}")
                logger.error(f"[Attempt {attempt + 1}/{max_retries}] Problematic JSON response: {json_str}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying ticker extraction (attempt {attempt + 2}/{max_retries})...")
        
        # All retries exhausted
        logger.error(f"All {max_retries} attempts failed for ticker extraction. Last error: {last_error}")
        return TickerExtractionOutput(items=[])
    
    async def extract_ticker_info_async(self, query: str, default_year: int, max_retries: int = 3) -> TickerExtractionOutput:
        """
        Asynchronous version of extract_ticker_info method.
        
        Args:
            query: The user query about financial reports
            default_year: Default year to use if no year is specified in the query
            max_retries: Maximum number of retry attempts if JSON parsing fails
            
        Returns:
            TickerExtractionOutput containing list of ticker-year pairs
        """
        # Format system prompt with default year
        system_prompt = TICKER_EXTRACTION_SYSTEM_PROMPT.format(default_year=default_year)
        
        updated_messages = [
            {
                "role": MessageRole.SYSTEM.value,
                "content": system_prompt
            },
            {
                "role": MessageRole.USER.value,
                "content": TICKER_EXTRACTION_USER_PROMPT.format(query=query),
            },
        ]

        inference_parameters = InferenceParameters(
            response_format={"type": "json_object"}
        )
        
        last_error = None
        last_json_str = None
        
        for attempt in range(max_retries):
            json_str = await self.inference_facade.create_chat_completion_async(
                messages=updated_messages,
                inference_parameters=inference_parameters
            )
            last_json_str = json_str
            
            # Try to parse JSON response
            try:
                result = self._parse_ticker_extraction_response(json_str)
                if result is not None:
                    logger.debug(f"Ticker extraction output (async): {result}")
                    return result
            except (json.JSONDecodeError, TypeError, KeyError, ValueError) as e:
                last_error = e
                # Always log problematic JSON immediately (not just in verbose mode)
                logger.error(f"[Attempt {attempt + 1}/{max_retries}] Failed to parse ticker extraction JSON: {e}")
                logger.error(f"[Attempt {attempt + 1}/{max_retries}] Problematic JSON response: {json_str}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying ticker extraction (attempt {attempt + 2}/{max_retries})...")
        
        # All retries exhausted
        logger.error(f"All {max_retries} attempts failed for ticker extraction. Last error: {last_error}")
        return TickerExtractionOutput(items=[])
    
    def _parse_ticker_extraction_response(self, json_str: str) -> Optional[TickerExtractionOutput]:
        """
        Parse the JSON response from ticker extraction.
        
        Args:
            json_str: The JSON string response from the model
            
        Returns:
            TickerExtractionOutput if parsing succeeds, None if structure is unexpected
            
        Raises:
            json.JSONDecodeError: If JSON is malformed
            TypeError, KeyError, ValueError: If JSON structure is invalid
        """
        parsed_json = json.loads(json_str)
        
        # Handle case where model returns object with array inside
        if isinstance(parsed_json, dict):
            # Try common keys that might contain the array
            for key in ['tickers', 'items', 'data', 'results']:
                if key in parsed_json and isinstance(parsed_json[key], list):
                    parsed_json = parsed_json[key]
                    break
            else:
                # If it's a single item dict with ticker_symbol and years, wrap in list
                if 'ticker_symbol' in parsed_json and 'years' in parsed_json:
                    parsed_json = [parsed_json]
                else:
                    # Unexpected structure - raise to trigger retry
                    raise ValueError(f"Unexpected JSON structure: {parsed_json}")
        
        if not isinstance(parsed_json, list):
            raise ValueError(f"Expected list but got {type(parsed_json)}: {parsed_json}")
        
        # Validate and create output
        output = TickerExtractionOutput.from_json_list(parsed_json)
        
        # Validate ticker symbols (basic validation - all caps, reasonable length)
        validated_items = []
        for item in output.items:
            if self._validate_ticker_symbol(item.ticker_symbol):
                validated_items.append(item)
            else:
                logger.warning(f"Invalid ticker symbol: {item.ticker_symbol}")
        
        output.items = validated_items
        return output
    
    def _validate_ticker_symbol(self, ticker: str) -> bool:
        """
        Validate a ticker symbol.
        
        Args:
            ticker: The ticker symbol to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not ticker:
            return False
        
        # Must be all uppercase letters (and possibly numbers for some tickers)
        if not re.match(r'^[A-Z0-9]+$', ticker):
            return False
        
        # Reasonable length (1-5 characters for US stocks)
        if len(ticker) < 1 or len(ticker) > 5:
            return False
        
        return True