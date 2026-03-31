from __future__ import annotations

import argparse

from app.clients import get_openai_client
from app.config import settings

SYSTEM_PROMPT = (
    "You are a cybersecurity assistant for 911 call centers and PSAPs. "
    "Stay within cyber risk, emergency communications, public safety operations, and related policy."
)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=settings.openai_finetuned_model or settings.openai_chat_model)
    args = parser.parse_args()
    client = get_openai_client()

    print("Fine-tuned model chat. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        response = client.responses.create(
            model=args.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
        )
        print(f"Assistant: {response.output_text.strip()}\n")


if __name__ == "__main__":
    main()
