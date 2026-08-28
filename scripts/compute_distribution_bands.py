"""Compute Low/Mid/High score band percentages for all 6 systems on the
full 1500-query FinDER benchmark.

Bands match the paper's Table 2 (distribution_shift) definition:
  Low:    s <= 3
  Medium: 4 <= s <= 7
  High:   s >= 8

Outputs to stdout (CSV-formatted) so it can be pasted into the LaTeX table.
"""
import csv
import glob
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"

GROUPS = ["01", "02", "03", "04", "05"]

PATTERNS = {
    "CBR":             "benchmark-Linq-AI-Research_FinDER-index-solution-Linq-AI-Research_FinDER-300-{g}-*.csv",
    "SFR":             "benchmark-Linq-AI-Research_FinDER-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
    "Agentic":         "benchmark-Linq-AI-Research_FinDER-agentic-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
    "CBR+Meta (V-A)":  "benchmark-Linq-AI-Research_FinDER-cbrmeta-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
    "CBR+LLM-Ctx (V-B)":"benchmark-Linq-AI-Research_FinDER-cbrllm-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
    "HDRR":            "benchmark-Linq-AI-Research_FinDER-hybrid-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
}


def resolve_one(system: str, group: str) -> str:
    abs_pattern = str(OUTPUT_DIR / PATTERNS[system].format(g=group))
    matches = sorted(glob.glob(abs_pattern))
    if system == "CBR":
        matches = [m for m in matches if "no-rerank" not in m]
    if not matches:
        raise FileNotFoundError(abs_pattern)
    return matches[-1]


def load_scores(path: str) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("score") or "").strip()
            try:
                s = int(raw)
            except (ValueError, TypeError):
                continue
            if 1 <= s <= 10:
                out.append(s)
    return out


def bands(scores):
    n = len(scores)
    if n == 0:
        return (0.0, 0.0, 0.0, 0)
    low = sum(1 for s in scores if s <= 3) / n * 100
    mid = sum(1 for s in scores if 4 <= s <= 7) / n * 100
    high = sum(1 for s in scores if s >= 8) / n * 100
    return (low, mid, high, n)


def main():
    print(f"{'System':22s}{'n':>6s}{'Low(s<=3)':>12s}{'Mid(4-7)':>12s}{'High(s>=8)':>12s}")
    for system in PATTERNS:
        all_scores = []
        for g in GROUPS:
            path = resolve_one(system, g)
            all_scores.extend(load_scores(path))
        low, mid, high, n = bands(all_scores)
        # Sanity: bands should sum to ~100
        assert abs(low + mid + high - 100) < 0.01, f"{system}: bands sum to {low+mid+high}"
        print(f"{system:22s}{n:>6d}{low:>12.2f}{mid:>12.2f}{high:>12.2f}")


if __name__ == "__main__":
    main()
