# financial-RAG-pipeline

Reproducibility repository for **"Sustainable Hybrid Document-Routed Retrieval for Financial RAG: Resolving the Robustness–Precision Trade-off"** (*Intelligent Systems with Applications*, 2026).

This repository contains the full retrieval pipeline, every system compared in the paper, the fixed query splits, and the raw benchmark outputs behind each reported table.

---

## What the paper evaluates

Six retrieval systems are compared over 1,500 FinDER queries (five fixed groups of 300):

| System | Description | Entry point |
|---|---|---|
| **CBR** | Chunk-based retrieval: hybrid FAISS + BM25, RRF fusion, neural reranking | `main_query.py` |
| **SFR** | Structured file routing: LLM parses the query to a filename, whole document read | `main_filename_based_solution.py` |
| **HDRR** | Hybrid document-routed retrieval: SFR routing (stage 1) + CBR retrieval within routed docs (stage 2), with full-corpus fallback | `main_hybrid_query.py` |
| **CBR+Meta (V-A)** | CBR over an index with a static `[Company: TICKER, Fiscal Year: YEAR]` chunk prefix | `main_index_contextual.py --contextual metadata` |
| **CBR+LLM-Ctx (V-B)** | CBR over an index with per-chunk LLM-generated context | `main_index_contextual.py --contextual llm` |
| **Agentic** | Post-hoc verification + retry over CBR (Self-RAG pattern) | `main_agentic_query.py` |

Headline result: HDRR reaches 7.54 average score / 6.4% failure rate versus 6.02 / 22.5% for CBR, at exactly 2 LLM calls per query.

## Repository layout

```
├── src/                  Pipeline source (indexing, retrieval, routing, evaluation)
│   ├── run_indexing.py            CLI: build an index over a document folder
│   ├── run_query.py               CLI: ask questions
│   ├── constants.py               Tunables + .env loader
│   ├── main/                      Argparse entry points (one per system)
│   ├── beans/ dao/ service/       Embedder, reranker, parsers, search services
│   └── tests/
├── scripts/              Analysis + aggregation utilities used to build the tables
├── random_queries_csv/   The five fixed 300-query FinDER splits (G01–G05)
├── example_queries/      Small sample query CSV
├── output/               Raw per-query outputs and benchmark scores (112 CSVs)
└── docs/                 Published supplementary material (Tables S1–S3)
```

## Setup

Python 3.10+.

```bash
cd src
pip install -r requirements_query.txt      # indexing + query
# or
pip install -r requirements_indexing.txt   # indexing only
```

Copy `.env.template` to `.env` at the repository root and add your key:

```env
OPENAI_API_KEY=sk-...
JINA_API_KEY=...          # optional; only for cloud reranking
```

`src/constants.py` loads it automatically. Shell environment variables take precedence. A missing `OPENAI_API_KEY` raises a clear error at startup.

**Optional local reranker.** All experiments in the paper use a local GGUF reranker rather than the Jina cloud API. It is an optional extra:

```bash
pip install -r requirements_local_reranker.txt
```

Then place the model and tokenizer under `local_llms/`:

```
local_llms/jina-reranker-v2-base-multilingual-F16.gguf
local_llms/jina_rerank_tokenizer.json
```

**Known issue:** the pinned `nexaai==1.0.37rc9` fetches a platform binary at build time and that download currently returns HTTP 403, so this install may fail. The rest of the pipeline is unaffected — the import is lazy. Use `--rerank-jina` (with `JINA_API_KEY`) or `--no-rerank` in the meantime; note that reranking materially affects results, so neither reproduces the paper's numbers exactly.

**Windows note:** some files under `output/` have 133-character paths. Clone into a short directory (e.g. `C:\src\`), or enable long paths with `git config --global core.longpaths true`.

## Reproducing the paper

The corpus is 497 SEC 10-K filings (S&P 500, FY2023). It is not redistributed here — see *Data availability* below.

**1. Build the index** (`cwd = src/`):

```bash
python run_indexing.py /path/to/10k/pdfs --project-name full_10k
```

Produces `full_10k.rag_pipeline.db` and `full_10k.faiss_index.index`. The paper's index holds 169,640 chunks over 497 documents.

**2. Run a system over a fixed query group:**

```bash
# CBR
python -m main.main_query --csv ../random_queries_csv/Linq-AI-Research_FinDER-300-01.csv \
    -p full_10k --column query --id-column query_id \
    --output-csv ../output/cbr-group-01.csv --concurrent 3 --rerank-use-local

# HDRR
python -m main.main_hybrid_query --csv ../random_queries_csv/Linq-AI-Research_FinDER-300-01.csv \
    -p full_10k --data-dir /path/to/10k/pdfs --default-year 2023 \
    --column query --id-column query_id \
    --output-csv ../output/hdrr-group-01.csv --concurrent 3 --rerank-use-local
```

`scripts/*.ps1` hold the exact invocations used for each system.

**3. Score against FinDER ground truth:**

```bash
python -m main.main_benchmark_rag_result --input-csv ../output/hdrr-group-01.csv \
    --hf-dataset Linq-AI-Research/FinDER --gt-field answer --id-field _id \
    --split train --concurrent 3
```

Note `--id-field _id` — that is FinDER's HuggingFace field name, not the `query_id` column in our CSVs.

**4. Aggregate:**

```bash
python scripts/compute_all_systems_per_group.py
```

This regenerates `output/all_systems_per_group_metrics.csv`, which is Table S1 of the supplementary material and whose mean rows are Table 6 of the article.

## How it works

**Indexing** is two-stage: a CPU-bound parse + chunk pass (`ProcessPoolExecutor`, `pypdfium2`) writing to SQLite with an FTS5 virtual table kept in sync by triggers, then an I/O-bound embed pass (`asyncio`) filling a FAISS `IndexIDMap2`.

**Querying** is rewrite → hybrid retrieval → rerank → answer. Retrieval merges FAISS semantic search with SQLite FTS5 BM25 via Reciprocal Rank Fusion (`utils/rrf.py`). Reranking uses a cumulative-probability threshold plus a cliff cutoff to decide how many chunks survive.

**Document routing** (SFR/HDRR) parses the query into `{ticker, year}` with structured output, resolves it against the filing repository laid out as `{year}/{ticker}.pdf`, and — in HDRR — constrains stage-2 chunk retrieval to the routed documents, falling back to full-corpus retrieval when routing fails.

### Defaults (`src/constants.py`)

| Setting | Value |
|---|---|
| Chunk size / overlap | 2500 / 1250 |
| Embedding model / dim / batch | `text-embedding-3-small` / 1024 / 32 |
| Semantic / FTS retrieval limit | 30 / 20 |
| RRF k | 60 |
| Final chunks assembled | 10 |
| Rerank keep threshold / cliff cutoff | 0.45 / 0.15 |

## Results

`output/` holds the raw artifacts, named `{system}-{dataset}-{group}-{model}-{timestamp}.csv`:

- `*-300-0N-*.csv` — per-query outputs (retrieved chunk ids, routed documents, generated answer)
- `benchmark-*.csv` — per-query LLM-as-judge scores against FinDER ground truth
- `all_systems_per_group_metrics.csv` — the aggregate table
- `cross_company_coverage_*.csv` — the 25-query multi-document evaluation
- `claude_eval_*.csv` — the cross-evaluator check (Claude scoring the same answers)

`docs/supplementary.pdf` is the supplementary material as published.

## Data availability

The 10-K corpus is not redistributed. Filings are public and retrievable from [SEC EDGAR](https://www.sec.gov/edgar); the ticker set is recoverable from the `document_ids` and `routed_documents` columns in `output/`. Queries and ground-truth answers come from [`Linq-AI-Research/FinDER`](https://huggingface.co/datasets/Linq-AI-Research/FinDER) on HuggingFace; the five fixed 300-query splits used throughout the paper are in `random_queries_csv/`.

## Tests

```bash
cd src
python -m unittest discover tests/db_tests
```

55 tests, no API key or corpus required.

## Citation

```bibtex
@article{cheng2026hdrr,
  title   = {Sustainable Hybrid Document-Routed Retrieval for Financial RAG:
             Resolving the Robustness--Precision Trade-off},
  author  = {Cheng, Zhiyuan and Lai, Longying and Liu, Yue},
  journal = {Intelligent Systems with Applications},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
