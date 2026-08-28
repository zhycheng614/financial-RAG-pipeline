#!/usr/bin/env python
"""
Cross-Company Query Auditor (Phase 4 / Experiment 4 step 1).

Scans the five FinDER 300-query CSVs for queries that mention two or more
distinct tickers from the indexed corpus. Reports counts and a candidate
list so the user can spot-check whether each query genuinely requires
cross-company comparison.

Usage:
    python scripts/audit_cross_company_queries.py \\
        --queries-dir ./random_queries_csv \\
        --db src/full_10k.rag_pipeline.db \\
        --report ./output/cross_company_audit.csv \\
        --multi-only-report ./output/cross_company_candidates.csv

Ticker detection: pulls the canonical ticker set from the indexed corpus
(`documents.file_name` minus `.pdf`), then case-sensitive substring-matches
each ticker as a standalone token in the query.
"""

import argparse
import csv
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

GROUPS = ["01", "02", "03", "04", "05"]


def load_ticker_set(db_path: Path) -> Set[str]:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("SELECT file_name FROM documents")
    tickers = {Path(r[0]).stem.upper() for r in cur.fetchall()}
    con.close()
    return {t for t in tickers if t}


def build_token_regex(tickers: Set[str]) -> re.Pattern:
    """Match each ticker as a standalone uppercase token: word boundary,
    capitals only, length 1-5. We sort by length descending to prefer
    longer matches when one ticker is a prefix of another."""
    sorted_tickers = sorted(tickers, key=len, reverse=True)
    escaped = [re.escape(t) for t in sorted_tickers]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b")


def find_tickers_in_query(query: str, regex: re.Pattern, tickers: Set[str]) -> List[str]:
    """Return the list of distinct ticker hits in the query."""
    hits = regex.findall(query)
    distinct: List[str] = []
    seen = set()
    for h in hits:
        if h in tickers and h not in seen:
            distinct.append(h)
            seen.add(h)
    return distinct


def load_query_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit FinDER query CSVs for cross-company (multi-ticker) queries."
        )
    )
    parser.add_argument(
        "--queries-dir",
        default="./random_queries_csv",
        help="Directory containing the 5 FinDER 300-query CSVs.",
    )
    parser.add_argument(
        "--csv-pattern",
        default="Linq-AI-Research_FinDER-300-{group}.csv",
        help="Filename pattern with {group} placeholder.",
    )
    parser.add_argument(
        "--db",
        default="src/full_10k.rag_pipeline.db",
        help="Source DB to pull the indexed ticker set from.",
    )
    parser.add_argument(
        "--report",
        default="./output/cross_company_audit.csv",
        help="Per-query report with tickers detected.",
    )
    parser.add_argument(
        "--multi-only-report",
        default="./output/cross_company_candidates.csv",
        help="Same report filtered to queries with >=2 distinct tickers.",
    )
    parser.add_argument(
        "--query-column",
        default="query",
        help="Column name holding the query text.",
    )
    parser.add_argument(
        "--id-column",
        default="query_id",
        help="Column name holding the query ID.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    queries_dir = Path(args.queries_dir).resolve()
    db_path = Path(args.db).resolve()

    if not queries_dir.is_dir():
        print(f"Error: queries-dir not a directory: {queries_dir}", file=sys.stderr)
        return 1
    if not db_path.exists():
        print(f"Error: db not found: {db_path}", file=sys.stderr)
        return 1

    tickers = load_ticker_set(db_path)
    print(f"Loaded {len(tickers)} tickers from {db_path}")
    regex = build_token_regex(tickers)

    all_rows: List[Dict[str, str]] = []
    multi_rows: List[Dict[str, str]] = []
    per_group_counts: Dict[str, Dict[str, int]] = {}
    multi_freq: Counter = Counter()  # how many queries each ticker appears in (multi-ticker only)
    pair_freq: Counter = Counter()   # how often each ticker pair co-occurs

    for group in GROUPS:
        path = queries_dir / args.csv_pattern.format(group=group)
        if not path.exists():
            print(f"  group {group}: missing -> {path}", file=sys.stderr)
            continue
        rows = load_query_csv(path)
        zero = single = multi = 0
        for r in rows:
            qid = r.get(args.id_column, "")
            qtext = r.get(args.query_column, "") or ""
            hits = find_tickers_in_query(qtext, regex, tickers)
            n = len(hits)
            row_out = {
                "group": group,
                "query_id": qid,
                "query": qtext,
                "n_tickers": str(n),
                "tickers": ",".join(hits),
            }
            all_rows.append(row_out)
            if n == 0:
                zero += 1
            elif n == 1:
                single += 1
            else:
                multi += 1
                multi_rows.append(row_out)
                for t in hits:
                    multi_freq[t] += 1
                for i in range(len(hits)):
                    for j in range(i + 1, len(hits)):
                        a, b = sorted((hits[i], hits[j]))
                        pair_freq[(a, b)] += 1
        per_group_counts[group] = {
            "total": len(rows),
            "zero_tickers": zero,
            "one_ticker": single,
            "multi_ticker": multi,
        }
        print(
            f"  group {group}: total={len(rows)}  zero={zero}  one={single}  multi={multi}"
        )

    # Write reports
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["group", "query_id", "query", "n_tickers", "tickers"]
        )
        writer.writeheader()
        writer.writerows(all_rows)
    with open(args.multi_only_report, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["group", "query_id", "query", "n_tickers", "tickers"]
        )
        writer.writeheader()
        writer.writerows(multi_rows)

    print()
    print("Summary across all 5 groups:")
    print(f"  Total queries:                {sum(c['total'] for c in per_group_counts.values())}")
    print(f"  Queries w/ 0 tickers matched: {sum(c['zero_tickers'] for c in per_group_counts.values())}")
    print(f"  Queries w/ 1 ticker matched:  {sum(c['one_ticker'] for c in per_group_counts.values())}")
    print(
        f"  Queries w/ >=2 tickers (cross-company candidates): "
        f"{sum(c['multi_ticker'] for c in per_group_counts.values())}"
    )

    if multi_rows:
        print()
        print(f"Top tickers appearing in multi-ticker queries (top 15):")
        for tic, freq in multi_freq.most_common(15):
            print(f"  {tic:<6}  {freq}")
        print()
        print(f"Top co-occurring ticker pairs (top 15):")
        for (a, b), freq in pair_freq.most_common(15):
            print(f"  {a:<6} + {b:<6}  {freq}")

    print()
    print(f"Per-query report  -> {args.report}")
    print(f"Multi-only report -> {args.multi_only_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
