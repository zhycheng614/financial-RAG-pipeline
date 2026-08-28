from typing import List, Optional
from dataclasses import dataclass


@dataclass
class RewriteServiceOutput:
    clarified_query: str
    keywords: List[str]