from typing import List
from dataclasses import dataclass


@dataclass
class TickerYearInfo:
    """Represents a ticker symbol with associated years."""
    ticker_symbol: str
    years: List[int]
    
    def __post_init__(self):
        """Validate ticker symbol format."""
        # Ensure ticker symbol is uppercase
        self.ticker_symbol = self.ticker_symbol.upper()
        
        # Ensure years is a list
        if not isinstance(self.years, list):
            self.years = [self.years]
        
        # Ensure all years are integers
        self.years = [int(year) for year in self.years]


@dataclass
class TickerExtractionOutput:
    """Output from ticker extraction service containing list of ticker-year pairs."""
    items: List[TickerYearInfo]
    
    @classmethod
    def from_json_list(cls, json_list: List[dict]) -> 'TickerExtractionOutput':
        """
        Create TickerExtractionOutput from a list of dictionaries.
        
        Args:
            json_list: List of dicts with 'ticker_symbol' and 'years' keys
            
        Returns:
            TickerExtractionOutput instance
        """
        items = []
        for item in json_list:
            if 'ticker_symbol' in item and 'years' in item:
                items.append(TickerYearInfo(
                    ticker_symbol=item['ticker_symbol'],
                    years=item['years']
                ))
        return cls(items=items)
    
    def get_file_paths(self, base_dir: str) -> List[str]:
        """
        Generate file paths for all ticker-year combinations.
        
        Args:
            base_dir: Base directory containing year subdirectories
            
        Returns:
            List of file paths in format: {base_dir}/{year}/{ticker_symbol}.pdf
        """
        import os
        
        paths = []
        for item in self.items:
            for year in item.years:
                path = os.path.join(base_dir, str(year), f"{item.ticker_symbol}.pdf")
                paths.append(path)
        return paths
    
    def is_empty(self) -> bool:
        """Check if no ticker symbols were extracted."""
        return len(self.items) == 0
