import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Console encoding
# ---------------------------------------------------------------------------
# The CLIs print status symbols (checkmarks, warning signs). On a default
# Windows console stdout is cp1252 and those raise UnicodeEncodeError mid-run,
# so force UTF-8 where the stream supports it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# ---------------------------------------------------------------------------
# Secrets / environment loading
# ---------------------------------------------------------------------------
# The .env file lives at the actual project root (one level above src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def _load_env_file(env_path: Path) -> None:
    """Minimal .env loader: KEY=VALUE per line, '#' comments, optional quotes.

    Existing environment variables take precedence over .env entries so a shell
    export can still override the file.
    """
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip surrounding single or double quotes if present
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(ENV_FILE)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            f"Add it to {ENV_FILE} (see .env.template) or export it in your shell."
        )
    return value


# Secrets — resolved lazily through env vars so they're never committed to source.
OPENAI_API_KEY = _require_env("OPENAI_API_KEY")
JINA_API_KEY = os.environ.get("JINA_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()


# Chunking constants
DEFAULT_CHUNK_SIZE = 2500
DEFAULT_CHUNK_OVERLAP = 1250

# PDF parser constants


# Embedding constants
EMBEDDING_MAX_BATCH_SIZE = 32
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


# Reranking constants
DEFAULT_RERANKER_MODEL = "jina-reranker-v2-base-multilingual"
DEFAULT_RERANKER_URL = "https://api.jina.ai/v1/rerank"

# Local reranker constants
DEFAULT_LOCAL_RERANKER_MODEL_PATH = "local_llms/jina-reranker-v2-base-multilingual-F16.gguf"
DEFAULT_LOCAL_RERANKER_TOKENIZER_PATH = "local_llms/jina_rerank_tokenizer.json"
DEFAULT_LOCAL_RERANKER_NUM_WORKERS = 10

# RERANK CONFIGURATION
# ===================
DEFAULT_MAX_ITEM_TO_RERANK = 30
DEFAULT_RERANK_KEEP_THRESHOLD = 0.45  # 1 - 0.45 = 0.55, 0.55 is the cumulative probability threshold for reranking
DEFAULT_RERANK_CLIFF_CUTOFF_SCORE_DIFFERENCE = 0.15  # If the score difference from the first score exceeds this threshold, cutoff
DEFAULT_RERANK_N_CTX = 4096
DEFAULT_RERANK_N_BATCH = 4096


# Retriever constants
DEFAULT_SEMANTIC_FILTERING_THRESHOLD = 2.0
DEFAULT_FINAL_CHUNKS_TO_ASSEMBLE_LIMIT = 10
DEFAULT_FTS_RETRIEVAL_LIMIT = 20
DEFAULT_SEMANTIC_RETRIEVAL_LIMIT = 30
DEFAULT_RRF_K = 60
