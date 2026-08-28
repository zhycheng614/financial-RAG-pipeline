#!/usr/bin/env python
"""
Refines the raw cross-company audit (`output/cross_company_candidates.csv`)
by filtering out false-positive matches where short single-letter or
abbreviation-like 2-letter tickers (e.g. D, A, T, CF, GM, IT, IR, MS, BK)
collide with common financial abbreviations such as "R&D", "SG&A",
"cash flow", "gross margin", "IT/IR".

A query is kept iff, after the abbreviation-like tickers are stripped,
the remaining matched-ticker set still has |tickers| >= 2.

Output:
    output/cross_company_candidates_refined.csv

Usage:
    python scripts/refine_cross_company_candidates.py
"""

from __future__ import annotations

import csv
from pathlib import Path

# Single-letter tickers are almost always abbreviation false positives in
# query text. 2-letter tickers below have been hand-checked against the
# FinDER query corpus as recurring noise:
#   CF  -> "cash flow"
#   GM  -> "gross margin"
#   IT  -> "information technology"
#   IR  -> "investor relations"
#   MS  -> "Morgan Stanley" sometimes, but also "Ms." / "MS&A"
#   BK  -> "bankruptcy"
#   NI  -> "net income"
#   EL  -> letter combos
#   J   -> "J." initials (covered by 1-char rule)
ABBREV_TICKERS = {
    # 1-char (drop all):
    "A", "C", "D", "F", "J", "K", "L", "O", "T", "V", "Z",
    # 2-char abbreviation collisions:
    "CF", "GM", "IT", "IR", "MS", "BK", "NI",
}

INPUT_CSV = Path("output/cross_company_candidates.csv")
OUTPUT_CSV = Path("output/cross_company_candidates_refined.csv")


def main() -> int:
    if not INPUT_CSV.exists():
        print(f"missing {INPUT_CSV}")
        return 1

    refined_rows = []
    with INPUT_CSV.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tickers = [t.strip() for t in row["tickers"].split(",") if t.strip()]
            kept = [t for t in tickers if t not in ABBREV_TICKERS]
            if len(kept) >= 2:
                row["kept_tickers"] = ",".join(kept)
                row["n_kept"] = str(len(kept))
                refined_rows.append(row)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["group", "query_id", "query", "n_tickers", "tickers",
                      "n_kept", "kept_tickers"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(refined_rows)

    print(f"raw candidates:     {sum(1 for _ in INPUT_CSV.open(encoding='utf-8')) - 1}")
    print(f"refined candidates: {len(refined_rows)}")
    print(f"output:             {OUTPUT_CSV}")
    if refined_rows:
        print()
        print("Refined cross-company candidates:")
        for r in refined_rows:
            print(f"  [{r['group']}] {r['query_id']}  ({r['kept_tickers']})")
            print(f"      {r['query']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
