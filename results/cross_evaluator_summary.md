# Cross-evaluator correlation summary

Sample: 99 stratified queries (20 per group across 5 splits).

## Per-system results

| System | n | GPT-4.1 avg | Claude avg | Δ | Spearman ρ | Pearson r | Mean |Δ| | GPT-4.1 fail% | Claude fail% | GPT-4.1 corr≥7 | Claude corr≥7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CBR | 99 | 5.980 | 5.091 | -0.889 | 0.9348 | 0.9249 | 0.990 | 24.2% | 25.2% | 58.6% | 40.4% |
| SFR | 99 | 6.495 | 5.283 | -1.212 | 0.8818 | 0.8845 | 1.273 | 9.1% | 12.1% | 61.6% | 36.4% |
| HDRR | 99 | 7.071 | 6.141 | -0.929 | 0.8547 | 0.8888 | 1.152 | 11.1% | 10.1% | 69.7% | 56.6% |
| V-A (CBR+Meta) | 99 | 6.808 | 5.798 | -1.010 | 0.9066 | 0.8790 | 1.253 | 11.1% | 9.1% | 68.7% | 46.5% |
| V-B (CBR+LLM-Ctx) | 99 | 7.192 | 6.020 | -1.172 | 0.8515 | 0.8753 | 1.273 | 10.1% | 8.1% | 70.7% | 51.5% |
| Agentic | 99 | 6.374 | 5.434 | -0.939 | 0.9204 | 0.9277 | 1.061 | 20.2% | 19.2% | 60.6% | 45.5% |

## Cross-system ranking (by avg score, descending)

| Rank | GPT-4.1 | Claude |
|---:|---|---|
| 1 | V-B (CBR+LLM-Ctx) (7.192) | HDRR (6.141) |
| 2 | HDRR (7.071) | V-B (CBR+LLM-Ctx) (6.020) |
| 3 | V-A (CBR+Meta) (6.808) | V-A (CBR+Meta) (5.798) |
| 4 | SFR (6.495) | Agentic (5.434) |
| 5 | Agentic (6.374) | SFR (5.283) |
| 6 | CBR (5.980) | CBR (5.091) |

**Rank order preserved:** False
**System-level Spearman ρ (avg-by-system):** 0.8857