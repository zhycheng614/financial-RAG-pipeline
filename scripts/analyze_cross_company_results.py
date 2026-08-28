#!/usr/bin/env python
"""
Phase 4 / Experiment 4 — Cross-company result analysis (Lite mode, no GT).

For each of CBR, V-B, Agentic, HDRR running on the 25 handcrafted
cross-company queries, compute objective metrics that do NOT need a
written ground-truth answer:

  - retrieval_coverage = | named_tickers \cap retrieved_tickers | / | named_tickers |
        (named  = tickers explicitly mentioned in the query;
         retrieved = the set of tickers whose 10-K appears among the
                     document_ids in the system's output row.)

  - answer_coverage   = | named_tickers \cap mentioned_in_answer | / | named_tickers |

  - full_retrieval    = retrieval_coverage == 1.0
  - full_answer       = answer_coverage    == 1.0

For HDRR, also report routing_coverage from the routed_documents column.

For Agentic, also report verification_result distribution + avg retry count.

Output:
  output/cross_company_coverage_summary.csv  -- per-system aggregate
  output/cross_company_coverage_per_query.csv -- per-query per-system detail
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Set, Tuple

DB_PATH = Path("src/full_10k.rag_pipeline.db")
QUERIES_CSV = Path("random_queries_csv/Linq-AI-Research_FinDER-cross-company-handcrafted.csv")

SYSTEMS: List[Tuple[str, Path]] = [
    ("CBR",     Path("output/Linq-AI-Research_FinDER-cross-company-handcrafted-cbr-gpt-4.1.csv")),
    ("V-B",     Path("output/cbrllm-Linq-AI-Research_FinDER-cross-company-handcrafted-gpt-4.1.csv")),
    ("Agentic", Path("output/agentic-Linq-AI-Research_FinDER-cross-company-handcrafted-gpt-4.1.csv")),
    ("HDRR",    Path("output/hybrid-Linq-AI-Research_FinDER-cross-company-handcrafted-gpt-4.1.csv")),
]


def load_ticker_set(db_path: Path) -> Tuple[Set[str], Dict[str, str]]:
    """Return (ticker_set, doc_id_to_ticker)."""
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("SELECT id, file_name FROM documents")
    rows = cur.fetchall()
    con.close()
    tickers: Set[str] = set()
    doc_id_to_ticker: Dict[str, str] = {}
    for doc_id, fname in rows:
        t = Path(fname).stem.upper()
        tickers.add(t)
        doc_id_to_ticker[str(doc_id)] = t
    return tickers, doc_id_to_ticker


# Tokens that match real tickers in the corpus but are noise in the
# handcrafted query text. `D` collides with "R&D"; `K` collides with "10-K".
# Any other genuine 1-char tickers we deliberately used as named comparators
# (e.g. F = Ford in cc000018) must NOT be in this list.
NOISE_TOKENS: Set[str] = {"D", "K"}


def extract_named_tickers(query: str, all_tickers: Set[str]) -> List[str]:
    """Token-match uppercase ticker symbols in the query, skipping known
    abbreviation collisions (e.g. `D` from `R&D`, `K` from `10-K`)."""
    hits: List[str] = []
    seen: Set[str] = set()
    for tok in re.findall(r"\b[A-Z]{1,5}\b", query):
        if tok in all_tickers and tok not in seen and tok not in NOISE_TOKENS:
            hits.append(tok)
            seen.add(tok)
    return hits


def tickers_from_doc_ids(doc_ids_str: str, doc_id_to_ticker: Dict[str, str]) -> Set[str]:
    if not doc_ids_str:
        return set()
    out: Set[str] = set()
    for raw in doc_ids_str.split(","):
        raw = raw.strip()
        if raw and raw in doc_id_to_ticker:
            out.add(doc_id_to_ticker[raw])
    return out


def tickers_from_routed_paths(routed: str) -> Set[str]:
    """For HDRR's routed_documents column: comma-separated full paths."""
    if not routed:
        return set()
    out: Set[str] = set()
    for p in routed.split(","):
        p = p.strip()
        if not p:
            continue
        out.add(Path(p).stem.upper())
    return out


def tickers_in_answer(answer: str, named: List[str]) -> Set[str]:
    """A named ticker counts as 'in answer' iff it appears as a standalone
    uppercase token in the answer text."""
    found: Set[str] = set()
    for t in named:
        if re.search(r"\b" + re.escape(t) + r"\b", answer):
            found.add(t)
    return found


def coverage(named: List[str], present: Set[str]) -> float:
    if not named:
        return 0.0
    return len(set(named) & present) / len(named)


def load_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".",
                        help="Repo root (where src/, output/, random_queries_csv/ live).")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    db_path = (root / DB_PATH).resolve()
    queries_csv = (root / QUERIES_CSV).resolve()

    if not db_path.exists():
        print(f"DB missing: {db_path}", file=sys.stderr)
        return 1
    if not queries_csv.exists():
        print(f"Queries CSV missing: {queries_csv}", file=sys.stderr)
        return 1

    all_tickers, doc_id_to_ticker = load_ticker_set(db_path)
    print(f"Loaded {len(all_tickers)} tickers from corpus")

    # Load queries -> named tickers
    queries: Dict[str, Dict[str, object]] = {}
    for r in load_csv_dicts(queries_csv):
        qid = r["query_id"]
        q = r["query"]
        named = extract_named_tickers(q, all_tickers)
        queries[qid] = {"query": q, "named": named}

    print(f"Loaded {len(queries)} handcrafted queries")
    print()

    per_query_rows: List[Dict[str, str]] = []
    summary_rows: List[Dict[str, object]] = []

    for sys_name, csv_path in SYSTEMS:
        full = (root / csv_path).resolve()
        if not full.exists():
            print(f"  [{sys_name}] missing -> {full}", file=sys.stderr)
            continue
        rows = load_csv_dicts(full)
        retr_cov: List[float] = []
        ans_cov: List[float] = []
        route_cov: List[float] = []
        full_retr = 0
        full_ans = 0
        full_route = 0
        agentic_verif = Counter()
        agentic_retries: List[int] = []
        agentic_calls: List[int] = []
        agentic_elapsed: List[int] = []
        errors = 0

        for r in rows:
            qid = r.get("query_id", "").strip()
            q_named: List[str] = queries.get(qid, {}).get("named", [])  # type: ignore[assignment]
            if not q_named:
                # Should not happen for handcrafted set; skip.
                continue

            err = (r.get("error") or "").strip()
            if err:
                errors += 1
                retr_cov.append(0.0)
                ans_cov.append(0.0)
                if sys_name == "HDRR":
                    route_cov.append(0.0)
                continue

            retrieved = tickers_from_doc_ids(r.get("document_ids", "") or "", doc_id_to_ticker)
            answer = (r.get("final_answer") or "")
            in_ans = tickers_in_answer(answer, q_named)
            r_cov = coverage(q_named, retrieved)
            a_cov = coverage(q_named, in_ans)
            retr_cov.append(r_cov)
            ans_cov.append(a_cov)
            if r_cov == 1.0: full_retr += 1
            if a_cov == 1.0: full_ans += 1

            row_out = {
                "system": sys_name,
                "query_id": qid,
                "named_tickers": ",".join(q_named),
                "retrieved_tickers": ",".join(sorted(retrieved & set(q_named))),
                "missing_retrieved": ",".join(sorted(set(q_named) - retrieved)),
                "answer_mentions": ",".join(sorted(in_ans)),
                "missing_in_answer": ",".join(sorted(set(q_named) - in_ans)),
                "retrieval_coverage": f"{r_cov:.3f}",
                "answer_coverage": f"{a_cov:.3f}",
            }

            if sys_name == "HDRR":
                routed = tickers_from_routed_paths(r.get("routed_documents", "") or "")
                rt_cov = coverage(q_named, routed)
                route_cov.append(rt_cov)
                if rt_cov == 1.0: full_route += 1
                row_out["routed_tickers"] = ",".join(sorted(routed & set(q_named)))
                row_out["missing_routed"] = ",".join(sorted(set(q_named) - routed))
                row_out["routing_coverage"] = f"{rt_cov:.3f}"
                row_out["routing_status"] = r.get("routing_status", "")
            if sys_name == "Agentic":
                vr = (r.get("verification_result") or "").strip()
                agentic_verif[vr] += 1
                try: agentic_retries.append(int(r.get("retry_count") or "0"))
                except ValueError: pass
                try: agentic_calls.append(int(r.get("total_llm_calls") or "0"))
                except ValueError: pass
                try: agentic_elapsed.append(int(r.get("elapsed_ms") or "0"))
                except ValueError: pass
                row_out["verification_result"] = vr
                row_out["retry_count"] = r.get("retry_count", "")

            per_query_rows.append(row_out)

        n = len(rows)
        summary: Dict[str, object] = {
            "system": sys_name,
            "n_queries": n,
            "errors": errors,
            "avg_retrieval_coverage": round(mean(retr_cov), 3) if retr_cov else 0.0,
            "full_retrieval_n": full_retr,
            "full_retrieval_pct": f"{full_retr/n*100:.1f}%" if n else "0.0%",
            "avg_answer_coverage": round(mean(ans_cov), 3) if ans_cov else 0.0,
            "full_answer_n": full_ans,
            "full_answer_pct": f"{full_ans/n*100:.1f}%" if n else "0.0%",
        }
        if sys_name == "HDRR" and route_cov:
            summary["avg_routing_coverage"] = round(mean(route_cov), 3)
            summary["full_routing_n"] = full_route
            summary["full_routing_pct"] = f"{full_route/n*100:.1f}%" if n else "0.0%"
        if sys_name == "Agentic":
            summary["agentic_verif_dist"] = dict(agentic_verif)
            summary["avg_retry_count"] = round(mean(agentic_retries), 2) if agentic_retries else 0.0
            summary["avg_total_llm_calls"] = round(mean(agentic_calls), 2) if agentic_calls else 0.0
            summary["avg_elapsed_ms"] = int(mean(agentic_elapsed)) if agentic_elapsed else 0
        summary_rows.append(summary)

    # Pretty-print summary
    print()
    print("=" * 90)
    print("CROSS-COMPANY COVERAGE SUMMARY")
    print("=" * 90)
    for s in summary_rows:
        print()
        print(f"  System: {s['system']}")
        for k, v in s.items():
            if k == "system": continue
            print(f"    {k:30s} = {v}")

    # Write outputs
    out_dir = (root / "output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "cross_company_coverage_summary.csv"
    per_query_csv = out_dir / "cross_company_coverage_per_query.csv"

    summary_fields = [
        "system", "n_queries", "errors",
        "avg_retrieval_coverage", "full_retrieval_n", "full_retrieval_pct",
        "avg_answer_coverage", "full_answer_n", "full_answer_pct",
        "avg_routing_coverage", "full_routing_n", "full_routing_pct",
        "agentic_verif_dist", "avg_retry_count", "avg_total_llm_calls", "avg_elapsed_ms",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for s in summary_rows:
            row = {k: s.get(k, "") for k in summary_fields}
            if "agentic_verif_dist" in s and isinstance(s["agentic_verif_dist"], dict):
                row["agentic_verif_dist"] = json.dumps(s["agentic_verif_dist"])
            w.writerow(row)

    per_query_fields_base = [
        "system", "query_id", "named_tickers",
        "retrieved_tickers", "missing_retrieved",
        "answer_mentions", "missing_in_answer",
        "retrieval_coverage", "answer_coverage",
        "routed_tickers", "missing_routed", "routing_coverage", "routing_status",
        "verification_result", "retry_count",
    ]
    with per_query_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_query_fields_base)
        w.writeheader()
        for row in per_query_rows:
            w.writerow({k: row.get(k, "") for k in per_query_fields_base})

    print()
    print(f"Summary    -> {summary_csv}")
    print(f"Per-query  -> {per_query_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
