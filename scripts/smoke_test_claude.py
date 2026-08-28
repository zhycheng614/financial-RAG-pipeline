"""One-call sanity check for the AnthropicInferenceFacade wiring."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from beans.anthropic_inference_facade import AnthropicInferenceFacade
from data_classes.inference_parameters import InferenceParameters


async def main():
    facade = AnthropicInferenceFacade(model="claude-sonnet-4-6")
    response = await facade.create_chat_completion_async(
        [
            {"role": "system", "content": "Reply with JSON only: {\"score\": <int 1-10>}."},
            {"role": "user", "content": "Score the answer '4' for the question '2+2='. Output JSON."},
        ],
        inference_parameters=InferenceParameters(
            temperature=0.0, max_tokens=50,
            response_format={"type": "json_object"},
        ),
    )
    print("RAW:", repr(response))


if __name__ == "__main__":
    asyncio.run(main())
