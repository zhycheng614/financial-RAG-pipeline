"""Phase 5 — cross-evaluator correlation analysis.

For each of the six systems:
  - Load GPT-4.1 benchmark CSVs (5 per system, joined to the 99-query sample).
  - Load the Claude Sonnet 4.6 eval CSV (already 99 rows).
  - Join on query_id.
  - Compute Spearman rho, Pearson r, mean absolute difference,
    per-evaluator averages, and per-evaluator failure / correctness rates.

Also rebuild the cross-system avg-score ranking under both evaluators and
report whether the rank order is preserved.

Outputs:
  output/cross_evaluator_per_system.csv  -- one row per system
  output/cross_evaluator_per_query.csv   -- one row per (system, query_id)
  output/cross_evaluator_summary.md      -- human-readable summary

Spearman / Pearson are implemented in pure Python (scipy is not installed in
this env). Ties in Spearman use the standard average-rank convention.
"""
import csv
import glob
import math
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"

GROUPS = ["01", "02", "03", "04", "05"]

# IMPORTANT label convention — see compute_all_systems_per_group.py for the
# verification trail. `index-solution-*` is CBR (chunk-based), bare
# `Linq-AI-Research_FinDER-*` is SFR (whole-doc routing). Earlier versions of
# this script had these two patterns swapped.
GPT41_BENCHMARK_PATTERNS = {
    "cbr":     "benchmark-Linq-AI-Research_FinDER-index-solution-Linq-AI-Research_FinDER-300-{g}-*.csv",
    "sfr":     "benchmark-Linq-AI-Research_FinDER-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
    "hdrr":    "benchmark-Linq-AI-Research_FinDER-hybrid-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
    "cbrmeta": "benchmark-Linq-AI-Research_FinDER-cbrmeta-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
    "cbrllm":  "benchmark-Linq-AI-Research_FinDER-cbrllm-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
    "agentic": "benchmark-Linq-AI-Research_FinDER-agentic-Linq-AI-Research_FinDER-300-{g}-1-to-300-gpt-4.1-*.csv",
}

# Pretty labels used in the markdown summary.
SYSTEM_LABELS = {
    "cbr":     "CBR",
    "sfr":     "SFR",
    "hdrr":    "HDRR",
    "cbrmeta": "V-A (CBR+Meta)",
    "cbrllm":  "V-B (CBR+LLM-Ctx)",
    "agentic": "Agentic",
}


def load_sample_ids() -> set:
    with (OUTPUT_DIR / "cross_eval_sample_100.csv").open(encoding="utf-8") as f:
        return {row["query_id"] for row in csv.DictReader(f)}


def resolve_one(pattern: str) -> str:
    abs_pattern = str(OUTPUT_DIR / pattern)
    matches = sorted(glob.glob(abs_pattern))
    # The CBR benchmark glob `*index-solution-*` accidentally matches
    # `*no-rerank-index-solution-*` too. Drop the latter explicitly.
    matches = [m for m in matches if "no-rerank-index-solution" not in m]
    if not matches:
        raise FileNotFoundError(abs_pattern)
    return matches[-1]


def load_scores(paths, sample_ids):
    """Load query_id -> int score from a list of benchmark CSVs (one of `paths`).

    Skips rows without a valid integer score; restricted to `sample_ids`.
    """
    out = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qid = row.get("query_id", "")
                if qid not in sample_ids:
                    continue
                raw = (row.get("score") or "").strip()
                try:
                    score = int(raw)
                except (ValueError, TypeError):
                    continue
                if 1 <= score <= 10:
                    out[qid] = score
    return out


def average_ranks(values):
    """Return ranks using the average-tie convention. Lower index = lower rank."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-based ranks: positions i..j inclusive
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(sxx * syy)
    if denom == 0:
        return float("nan")
    return sxy / denom


def spearman(xs, ys):
    if len(xs) < 2:
        return float("nan")
    return pearson(average_ranks(xs), average_ranks(ys))


def fail_rate(scores):
    return sum(1 for s in scores if s == 1) / len(scores)


def correct_rate(scores, thresh=7):
    return sum(1 for s in scores if s >= thresh) / len(scores)


def perfect_rate(scores):
    return sum(1 for s in scores if s == 10) / len(scores)


def main():
    sample_ids = load_sample_ids()
    print(f"Sample size: {len(sample_ids)} query_ids")

    per_system_rows = []
    per_query_rows = []

    for system, pat in GPT41_BENCHMARK_PATTERNS.items():
        gpt41_paths = [resolve_one(pat.format(g=g)) for g in GROUPS]
        gpt41 = load_scores(gpt41_paths, sample_ids)
        claude = load_scores([str(OUTPUT_DIR / f"claude_eval_{system}.csv")], sample_ids)

        joined_ids = sorted(set(gpt41) & set(claude))
        gxs = [gpt41[q] for q in joined_ids]
        cxs = [claude[q] for q in joined_ids]

        # Absolute diff per query for the long table
        for q, g, c in zip(joined_ids, gxs, cxs):
            per_query_rows.append({
                "system": system, "query_id": q,
                "gpt41_score": g, "claude_score": c, "abs_diff": abs(g - c),
            })

        if not gxs:
            print(f"  {system}: 0 joined rows — skipping correlation")
            continue

        row = {
            "system": system,
            "label": SYSTEM_LABELS[system],
            "n": len(gxs),
            "gpt41_avg": round(mean(gxs), 3),
            "claude_avg": round(mean(cxs), 3),
            "delta_avg": round(mean(cxs) - mean(gxs), 3),
            "spearman_rho": round(spearman(gxs, cxs), 4),
            "pearson_r":    round(pearson(gxs, cxs), 4),
            "mean_abs_diff": round(mean(abs(g - c) for g, c in zip(gxs, cxs)), 3),
            "gpt41_fail":  round(fail_rate(gxs), 4),
            "claude_fail": round(fail_rate(cxs), 4),
            "gpt41_correct": round(correct_rate(gxs), 4),
            "claude_correct": round(correct_rate(cxs), 4),
            "gpt41_perfect": round(perfect_rate(gxs), 4),
            "claude_perfect": round(perfect_rate(cxs), 4),
        }
        per_system_rows.append(row)

    # Write per-system CSV
    per_system_csv = OUTPUT_DIR / "cross_evaluator_per_system.csv"
    with per_system_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_system_rows[0].keys()),
                                 quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(per_system_rows)

    # Write per-query CSV
    per_query_csv = OUTPUT_DIR / "cross_evaluator_per_query.csv"
    with per_query_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "query_id", "gpt41_score",
                                                "claude_score", "abs_diff"],
                                 quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(per_query_rows)

    # Cross-system rank preservation
    gpt41_rank = sorted(per_system_rows, key=lambda r: -r["gpt41_avg"])
    claude_rank = sorted(per_system_rows, key=lambda r: -r["claude_avg"])

    lines = []
    lines.append("# Cross-evaluator correlation summary")
    lines.append("")
    lines.append(f"Sample: {len(sample_ids)} stratified queries (20 per group across 5 splits).")
    lines.append("")
    lines.append("## Per-system results")
    lines.append("")
    lines.append("| System | n | GPT-4.1 avg | Claude avg | Δ | Spearman ρ | Pearson r | Mean |Δ| | GPT-4.1 fail% | Claude fail% | GPT-4.1 corr≥7 | Claude corr≥7 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in per_system_rows:
        lines.append(
            f"| {r['label']} | {r['n']} | {r['gpt41_avg']:.3f} | {r['claude_avg']:.3f} | "
            f"{r['delta_avg']:+.3f} | {r['spearman_rho']:.4f} | {r['pearson_r']:.4f} | "
            f"{r['mean_abs_diff']:.3f} | {r['gpt41_fail']*100:.1f}% | {r['claude_fail']*100:.1f}% | "
            f"{r['gpt41_correct']*100:.1f}% | {r['claude_correct']*100:.1f}% |"
        )
    lines.append("")
    lines.append("## Cross-system ranking (by avg score, descending)")
    lines.append("")
    lines.append("| Rank | GPT-4.1 | Claude |")
    lines.append("|---:|---|---|")
    for i in range(len(gpt41_rank)):
        lines.append(f"| {i+1} | {gpt41_rank[i]['label']} ({gpt41_rank[i]['gpt41_avg']:.3f}) | "
                     f"{claude_rank[i]['label']} ({claude_rank[i]['claude_avg']:.3f}) |")
    gpt41_order = [r["system"] for r in gpt41_rank]
    claude_order = [r["system"] for r in claude_rank]
    rank_preserved = gpt41_order == claude_order
    lines.append("")
    lines.append(f"**Rank order preserved:** {rank_preserved}")
    if not rank_preserved:
        # Compute system-level rank Spearman
        sys_gpt = [r["gpt41_avg"] for r in per_system_rows]
        sys_cla = [r["claude_avg"] for r in per_system_rows]
        rho = spearman(sys_gpt, sys_cla)
        lines.append(f"**System-level Spearman ρ (avg-by-system):** {rho:.4f}")

    summary_md = OUTPUT_DIR / "cross_evaluator_summary.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWrote {per_system_csv}")
    print(f"Wrote {per_query_csv}")
    print(f"Wrote {summary_md}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
