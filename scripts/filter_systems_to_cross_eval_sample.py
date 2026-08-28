"""Phase 5 — filter each system's per-group RAG CSVs to the 100-query subset.

For every (system, group) pair the per-group RAG result CSV is read, rows whose
query_id is in the 100-query sample are kept, and the union per system is
written as ``output/cross_eval_input_{system}.csv`` with a standard schema.

Standard output schema:
  original_row, query_id, query, final_answer, error

Only the fields the benchmark actually uses (`query_id`, `query`,
`final_answer`, `error`) plus an `original_row` for traceability are kept.
"""
import csv
import glob
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"

SAMPLE_PATH = OUTPUT_DIR / "cross_eval_sample_100.csv"

# Each entry maps a system label to a list of glob patterns relative to output/.
# We iterate groups 01..05 so the union ends up with exactly 100 rows when the
# sample is fully covered by the system.
GROUPS = ["01", "02", "03", "04", "05"]

# IMPORTANT label convention (verified against per-CSV columns and the existing
# paper Table 1):
#  - CBR is `index-solution-*` (chunk-based; columns: chunk_ids, document_ids).
#    Produced by main_filename_based_solution.py.
#  - SFR is the bare `Linq-AI-Research_FinDER-300-XX-*-gpt-4.1-*.csv`
#    (whole-doc routing; columns: tickers, file_paths, missing_files).
#  Earlier versions of this script had these two swapped.
SYSTEM_PATTERNS = {
    "cbr": "index-solution-Linq-AI-Research_FinDER-300-{g}.csv",
    "sfr": "Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
    "hdrr": "hybrid-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
    "cbrmeta": "cbrmeta-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
    "cbrllm": "cbrllm-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
    "agentic": "agentic-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
}

OUTPUT_FIELDS = ["original_row", "query_id", "query", "final_answer", "error"]


def load_sample_ids() -> set:
    ids = set()
    with SAMPLE_PATH.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ids.add(row["query_id"])
    return ids


def resolve_one(pattern: str) -> str:
    """Return the single matching file; abort if zero or many match."""
    abs_pattern = str(OUTPUT_DIR / pattern)
    matches = sorted(glob.glob(abs_pattern))
    if not matches:
        raise FileNotFoundError(f"No file matched: {abs_pattern}")
    if len(matches) > 1:
        # Prefer the most recent (highest timestamp suffix) when duplicates exist.
        # Sorting alphabetically already does this for the ISO-style timestamps used here.
        return matches[-1]
    return matches[0]


def filter_system(system: str, pattern_tpl: str, sample_ids: set):
    rows_out = []
    seen_ids = set()
    for g in GROUPS:
        src = resolve_one(pattern_tpl.format(g=g))
        with open(src, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = row.get("query_id", "")
                if qid in sample_ids and qid not in seen_ids:
                    rows_out.append({
                        "original_row": row.get("original_row", ""),
                        "query_id": qid,
                        "query": row.get("query", ""),
                        "final_answer": row.get("final_answer", ""),
                        "error": row.get("error", ""),
                    })
                    seen_ids.add(qid)

    dst = OUTPUT_DIR / f"cross_eval_input_{system}.csv"
    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows_out)

    missing = sample_ids - seen_ids
    print(f"  {system:8s}  rows={len(rows_out):3d}  missing={len(missing):3d}  -> {dst.name}")
    return len(rows_out), missing


def main():
    sample_ids = load_sample_ids()
    print(f"Loaded {len(sample_ids)} sample query_ids from {SAMPLE_PATH.name}")
    print()

    any_missing = False
    for system, pattern_tpl in SYSTEM_PATTERNS.items():
        try:
            count, missing = filter_system(system, pattern_tpl, sample_ids)
            if missing:
                any_missing = True
                print(f"    {system}: first 3 missing ids -> {sorted(missing)[:3]}")
        except FileNotFoundError as e:
            print(f"  {system:8s}  ERROR  {e}")
            any_missing = True

    if any_missing:
        print()
        print("WARNING: some systems are missing one or more sample queries.")
        sys.exit(1)


if __name__ == "__main__":
    main()
