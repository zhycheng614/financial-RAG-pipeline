"""Diagnose why the Claude facade returns empty strings for some eval calls."""
import asyncio
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from constants import ANTHROPIC_API_KEY
from prompts import BENCHMARK_EVALUATION_SYSTEM_PROMPT, BENCHMARK_EVALUATION_USER_PROMPT


async def run_one(query: str, rag_answer: str, ground_truth: str, max_tokens: int):
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY, max_retries=2)
    user_prompt = BENCHMARK_EVALUATION_USER_PROMPT.format(
        query=query, ground_truth=ground_truth, rag_answer=rag_answer,
    )
    resp = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        temperature=0.0,
        system=BENCHMARK_EVALUATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    print(f"  max_tokens={max_tokens}: stop_reason={resp.stop_reason!r} "
          f"usage(in={resp.usage.input_tokens}, out={resp.usage.output_tokens})")
    for i, block in enumerate(resp.content):
        btype = getattr(block, "type", None)
        text = getattr(block, "text", "")
        preview = text.replace("\n", "\\n")[:200]
        print(f"    block[{i}].type={btype}, text={preview!r}")


async def main():
    # Pull the first failing row out of the existing eval CSV
    failing_row = None
    eval_csv = REPO_ROOT / "output" / "claude_eval_cbr.csv"
    with eval_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("evaluation_error") and "Parse error" in row["evaluation_error"]:
                failing_row = row
                break
    if not failing_row:
        print("No failing row found")
        return

    print(f"Reproducing failure for query_id={failing_row['query_id']}")
    print(f"  query: {failing_row['query'][:100]}")
    print()

    for mt in (100, 256, 1024):
        await run_one(
            failing_row["query"],
            failing_row["rag_answer"],
            failing_row["ground_truth"],
            max_tokens=mt,
        )


if __name__ == "__main__":
    asyncio.run(main())
