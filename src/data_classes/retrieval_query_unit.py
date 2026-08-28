from dataclasses import dataclass
from typing import List

@dataclass
class RetrievalQueryUnit:
    query: str
    keywords: List[str] = None
    
    def get_combined_query_string(self) -> str:
        if self.keywords:
            return self.query + "\n" + ", ".join(self.keywords)
        else:
            return self.query