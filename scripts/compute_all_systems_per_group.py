"""Compute per-group and overall metrics for all 6 systems at consistent thresholds.

Outputs a single CSV: output/all_systems_per_group_metrics.csv
Columns: system, group, n, avg, fail_pct, corr7_pct, corr8_pct, perfect_pct
The corr7 and corr8 columns let the paper switch between the legacy s>=8 and the
benchmark-rubric-aligned s>=7 thresholds without recomputing.
"""
import csv
import glob
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"

GROUPS = ["01", "02", "03", "04", "05"]

# IMPORTANT label convention (verified against per-CSV columns and the existing
# paper Table 1):
#  - CBR (chunk-based retrieval) is `index-solution-*` (columns: chunk_ids,
#    document_ids, final_answer). Produced by main_filename_based_solution.py.
#  - SFR (semantic file routing / whole-doc) is the bare
#    `Linq-AI-Research_FinDER-300-XX-*-gpt-4.1-*.csv` (columns: tickers,
#    file_paths, missing_files, final_answer). Produced by main_query.py against
#    the file-routing path.
#  Earlier scripts in this repo had these two swapped — do NOT regress.
PATTERNS = {
    "cbr":     "benchmark-Linq-AI-Research_FinDER-index-solution-Linq-AI-Research_FinDER-300-{g}-*.csv",
    "sfr":     "benchmark-Linq-AI-Research_FinDER-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
    "hdrr":    "benchmark-Linq-AI-Research_FinDER-hybrid-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
    "cbrmeta": "benchmark-Linq-AI-Research_FinDER-cbrmeta-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
    "cbrllm":  "benchmark-Linq-AI-Research_FinDER-cbrllm-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
    "agentic": "benchmark-Linq-AI-Research_FinDER-agentic-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
}

# CBR (index-solution-*) must EXCLUDE no-rerank-index-solution-*.
def resolve_one(system: str, group: str) -> str:
    abs_pattern = str(OUTPUT_DIR / PATTERNS[system].format(g=group))
    matches = sorted(glob.glob(abs_pattern))
    if system == "cbr":
        matches = [m for m in matches if "no-rerank" not in m]
    if not matches:
        raise FileNotFoundError(abs_pattern)
    return matches[-1]


def load_scores(path: str) -> list:
    scores = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("score") or "").strip()
            try:
                s = int(raw)
            except (ValueError, TypeError):
                continue
            if 1 <= s <= 10:
                scores.append(s)
    return scores


def stats(scores):
    n = len(scores)
    if n == 0:
        return {"n": 0, "avg": float("nan"), "fail": 0, "c7": 0, "c8": 0, "p10": 0}
    return {
        "n": n,
        "avg": sum(scores) / n,
        "fail": sum(1 for s in scores if s == 1) / n,
        "c7":   sum(1 for s in scores if s >= 7) / n,
        "c8":   sum(1 for s in scores if s >= 8) / n,
        "p10":  sum(1 for s in scores if s == 10) / n,
    }


def main():
    out_rows = []
    print(f"{'system':10s}{'grp':>5s}{'n':>6s}{'avg':>7s}{'fail%':>7s}{'c7%':>7s}{'c8%':>7s}{'p10%':>7s}")
    for system in PATTERNS:
        all_scores = []
        for g in GROUPS:
            path = resolve_one(system, g)
            scores = load_scores(path)
            all_scores.extend(scores)
            st = stats(scores)
            out_rows.append({"system": system, "group": g, **st})
            print(f"{system:10s}{g:>5s}{st['n']:>6d}{st['avg']:>7.3f}{st['fail']*100:>7.2f}{st['c7']*100:>7.2f}{st['c8']*100:>7.2f}{st['p10']*100:>7.2f}")
        st = stats(all_scores)
        # per-group avg/std for the system
        avgs = [r["avg"] for r in out_rows if r["system"] == system and r["group"] in GROUPS]
        std_avg = statistics.pstdev(avgs)
        out_rows.append({"system": system, "group": "mean", **st})
        print(f"{system:10s}{'mean':>5s}{st['n']:>6d}{st['avg']:>7.3f}{st['fail']*100:>7.2f}{st['c7']*100:>7.2f}{st['c8']*100:>7.2f}{st['p10']*100:>7.2f}  (std_avg={std_avg:.3f})")
        print()

    out_csv = OUTPUT_DIR / "all_systems_per_group_metrics.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "group", "n", "avg", "fail", "c7", "c8", "p10"])
        w.writeheader()
        for r in out_rows:
            w.writerow({
                "system": r["system"],
                "group": r["group"],
                "n": r["n"],
                "avg": round(r["avg"], 4),
                "fail": round(r["fail"], 4),
                "c7": round(r["c7"], 4),
                "c8": round(r["c8"], 4),
                "p10": round(r["p10"], 4),
            })
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
