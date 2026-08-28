from dataclasses import dataclass

@dataclass
class BM25Query:
    input: str
    query: str
    is_exact: bool = False
    q_length: int = 0
    i_length: int = 0
    r_length: int = 0