#!/usr/bin/env python
"""
Routing Fallback Analyzer (Phase 3 — Experiment 2 / Section 7.3)

Joins HDRR query-output CSVs (which contain `routing_status`) with their
corresponding benchmark CSVs (which contain `score`) on `query_id`, then
reports per-status frequency and accuracy metrics, plus a candidate-failure
table to seed the qualitative case studies required by R4-B.

Usage:
    python scripts/analyze_routing_fallback.py \\
        --output-dir ./output \\
        --report-csv  ./output/routing_fallback_summary.csv \\
        --joined-csv  ./output/routing_fallback_joined.csv

The script auto-discovers the five FinDER 300-query groups by matching
`hybrid-Linq-AI-Research_FinDER-300-0{N}-*.csv` against
`benchmark-Linq-AI-Research_FinDER-hybrid-Linq-AI-Research_FinDER-300-0{N}-*.csv`.
"""

import argparse
import csv
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

GROUPS = ["01", "02", "03", "04", "05"]

# Group all "*fallback*" rows that map to full-corpus retrieval under
# one aggregated bucket for the paper table, while still keeping the
# per-variant rows visible.
FALLBACK_VALUES = {
    "fallback_no_tickers",
    "fallback_not_in_index",
    "fallback_no_files_on_disk",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join HDRR query output CSVs with benchmark score CSVs and "
            "report per-routing-status accuracy."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory containing both the hybrid-* and benchmark-* CSVs.",
    )
    parser.add_argument(
        "--report-csv",
        default="./output/routing_fallback_summary.csv",
        help="Where to write the per-status summary table.",
    )
    parser.add_argument(
        "--joined-csv",
        default="./output/routing_fallback_joined.csv",
        help="Where to write the full joined query-level table.",
    )
    parser.add_argument(
        "--failure-csv",
        default="./output/routing_fallback_failure_candidates.csv",
        help="Where to write candidate failure rows (low score + fallback).",
    )
    parser.add_argument(
        "--failure-max-score",
        type=int,
        default=4,
        help=(
            "Rows scored <= this value on a non-exact_path status are kept "
            "as failure-case candidates."
        ),
    )
    return parser.parse_args()


def discover_pair(output_dir: Path, group: str) -> Tuple[Path, Path]:
    """Locate the (hybrid_query_csv, benchmark_csv) pair for one group."""
    hybrid_glob = str(
        output_dir
        / f"hybrid-Linq-AI-Research_FinDER-300-{group}-1-to-300-gpt-4.1-*.csv"
    )
    bench_glob = str(
        output_dir
        / (
            "benchmark-Linq-AI-Research_FinDER-"
            f"hybrid-Linq-AI-Research_FinDER-300-{group}-1-to-300-gpt-4.1-*.csv"
        )
    )
    hybrid_matches = sorted(glob.glob(hybrid_glob))
    bench_matches = sorted(glob.glob(bench_glob))
    if not hybrid_matches:
        raise FileNotFoundError(f"No hybrid CSV found matching {hybrid_glob}")
    if not bench_matches:
        raise FileNotFoundError(f"No benchmark CSV found matching {bench_glob}")
    # Newest run wins if multiple exist.
    return Path(hybrid_matches[-1]), Path(bench_matches[-1])


def load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def join_rows(
    hybrid_rows: List[Dict[str, str]],
    bench_rows: List[Dict[str, str]],
    group: str,
) -> List[Dict[str, str]]:
    """Inner-join on query_id, carrying group label, routing status, score."""
    bench_by_qid: Dict[str, Dict[str, str]] = {
        r["query_id"]: r for r in bench_rows if r.get("query_id")
    }
    out: List[Dict[str, str]] = []
    for h in hybrid_rows:
        qid = h.get("query_id")
        if not qid or qid not in bench_by_qid:
            continue
        b = bench_by_qid[qid]
        out.append(
            {
                "group": group,
                "query_id": qid,
                "query": h.get("query", ""),
                "tickers": h.get("tickers", ""),
                "routed_documents": h.get("routed_documents", ""),
                "routing_status": h.get("routing_status", "").strip(),
                "rag_answer": b.get("rag_answer", ""),
                "ground_truth": b.get("ground_truth", ""),
                "score": b.get("score", "").strip(),
                "evaluation_error": b.get("evaluation_error", ""),
            }
        )
    return out


def parse_score(s: str) -> Optional[int]:
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def compute_status_stats(joined: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """For each routing_status (and a special 'fallback_any' aggregate plus
    an 'ALL' row), compute count, %, average score, failure rate (score==1),
    correctness rate (score>=8), perfect rate (score==10)."""
    buckets: Dict[str, List[int]] = defaultdict(list)
    for row in joined:
        sc = parse_score(row["score"])
        if sc is None:
            continue
        buckets[row["routing_status"]].append(sc)
    # Aggregate fallback_*
    fallback_scores: List[int] = []
    for status, scores in buckets.items():
        if status in FALLBACK_VALUES:
            fallback_scores.extend(scores)
    if fallback_scores:
        buckets["fallback_any (aggregated)"] = fallback_scores
    # Overall row
    all_scores: List[int] = []
    for status, scores in buckets.items():
        if status == "fallback_any (aggregated)":
            continue
        all_scores.extend(scores)
    total = len(all_scores)

    summary: List[Dict[str, str]] = []
    order = sorted(
        buckets.keys(),
        key=lambda k: (k == "fallback_any (aggregated)", -len(buckets[k])),
    )
    for status in order:
        scores = buckets[status]
        n = len(scores)
        if n == 0:
            continue
        avg = sum(scores) / n
        failure_rate = sum(1 for s in scores if s == 1) / n
        correctness_rate = sum(1 for s in scores if s >= 8) / n
        perfect_rate = sum(1 for s in scores if s == 10) / n
        pct = (n / total * 100.0) if (status != "fallback_any (aggregated)" and total) else (
            n / total * 100.0 if total else 0.0
        )
        summary.append(
            {
                "routing_status": status,
                "count": str(n),
                "percent_of_total": f"{pct:.2f}",
                "avg_score": f"{avg:.3f}",
                "failure_rate": f"{failure_rate:.3f}",
                "correctness_rate": f"{correctness_rate:.3f}",
                "perfect_rate": f"{perfect_rate:.3f}",
            }
        )
    # All
    if total:
        avg = sum(all_scores) / total
        summary.append(
            {
                "routing_status": "ALL",
                "count": str(total),
                "percent_of_total": "100.00",
                "avg_score": f"{avg:.3f}",
                "failure_rate": f"{sum(1 for s in all_scores if s == 1) / total:.3f}",
                "correctness_rate": f"{sum(1 for s in all_scores if s >= 8) / total:.3f}",
                "perfect_rate": f"{sum(1 for s in all_scores if s == 10) / total:.3f}",
            }
        )
    return summary


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_status_table(summary: List[Dict[str, str]]) -> None:
    headers = [
        "routing_status",
        "count",
        "percent_of_total",
        "avg_score",
        "failure_rate",
        "correctness_rate",
        "perfect_rate",
    ]
    widths = {h: max(len(h), max((len(r[h]) for r in summary), default=0)) for h in headers}
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("  ".join("-" * widths[h] for h in headers))
    for r in summary:
        print("  ".join(r[h].ljust(widths[h]) for h in headers))


def select_failure_candidates(
    joined: List[Dict[str, str]], max_score: int
) -> List[Dict[str, str]]:
    """Pull low-scoring queries on non-exact_path statuses so a human can
    pick rebrandings / subsidiaries / ambiguous-reference examples from a
    bounded list. Sort by status (rare first) then ascending score."""
    out = [
        r
        for r in joined
        if r["routing_status"] != "exact_path"
        and (parse_score(r["score"]) is not None and parse_score(r["score"]) <= max_score)
    ]
    status_freq: Dict[str, int] = Counter(r["routing_status"] for r in joined)
    out.sort(
        key=lambda r: (
            status_freq.get(r["routing_status"], 0),
            parse_score(r["score"]) or 0,
            r["query_id"],
        )
    )
    return out


def main() -> int:
    args = parse_arguments()
    output_dir = Path(args.output_dir).resolve()
    if not output_dir.is_dir():
        print(f"Error: --output-dir does not exist: {output_dir}", file=sys.stderr)
        return 1

    print(f"Scanning {output_dir} for HDRR + benchmark CSV pairs ...")
    joined_all: List[Dict[str, str]] = []
    for group in GROUPS:
        try:
            hybrid_path, bench_path = discover_pair(output_dir, group)
        except FileNotFoundError as e:
            print(f"  group {group}: {e}", file=sys.stderr)
            continue
        print(f"  group {group}: {hybrid_path.name}  +  {bench_path.name}")
        hybrid_rows = load_csv(hybrid_path)
        bench_rows = load_csv(bench_path)
        joined = join_rows(hybrid_rows, bench_rows, group)
        joined_all.extend(joined)

    if not joined_all:
        print("No joined rows; aborting.", file=sys.stderr)
        return 1

    print(f"\nJoined {len(joined_all)} rows across {len(GROUPS)} groups.")

    summary = compute_status_stats(joined_all)
    print()
    print("Per-routing-status summary:")
    print_status_table(summary)

    write_csv(Path(args.joined_csv), joined_all)
    write_csv(Path(args.report_csv), summary)

    candidates = select_failure_candidates(joined_all, args.failure_max_score)
    write_csv(Path(args.failure_csv), candidates)

    print()
    print(f"Joined rows  -> {args.joined_csv}  ({len(joined_all)})")
    print(f"Summary      -> {args.report_csv}  ({len(summary)} rows)")
    print(
        f"Failure pool -> {args.failure_csv}  "
        f"({len(candidates)} candidates, score <= {args.failure_max_score})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
