from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.clients import get_openai_client
from app.config import settings
from app.services.chat_service import answer_question


def judge(question: str, gold: str, answer: str, citations_used: int) -> dict:
    prompt = f"""
You are grading a PSAP cybersecurity assistant answer.
Return strict JSON with keys:
- groundedness: integer 1-5
- completeness: integer 1-5
- usefulness: integer 1-5
- citation_quality: integer 1-5
- notes: string

Question: {question}
Gold reference answer: {gold}
Candidate answer: {answer}
Citation count: {citations_used}
""".strip()

    client = get_openai_client()
    response = client.responses.create(
        model=settings.openai_chat_model,
        input=prompt,
    )
    text = response.output_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "groundedness": None,
            "completeness": None,
            "usefulness": None,
            "citation_quality": None,
            "notes": text,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--mode", default="rag", choices=["base", "finetuned", "rag", "hybrid"])
    parser.add_argument("--model-override")
    parser.add_argument("--namespace", default=settings.pinecone_namespace)
    parser.add_argument("--output-csv", default="data/staged/eval_results.csv")
    args = parser.parse_args()

    rows = []

    with Path(args.eval_file).open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            question = item["question"]
            gold = item.get("gold_answer", "")
            filters = item.get("filters")

            result = answer_question(
                question=question,
                namespace=args.namespace,
                filters=filters,
                mode=args.mode,
                model_override=args.model_override,
            )

            grade = judge(question, gold, result.answer, len(result.citations))

            rows.append(
                {
                    "mode": args.mode,
                    "chat_model_used": result.chat_model_used,
                    "question": question,
                    "gold_answer": gold,
                    "model_answer": result.answer,
                    "citation_count": len(result.citations),
                    **grade,
                }
            )

    df = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"Wrote eval results to {args.output_csv}")
    print(df[["groundedness", "completeness", "usefulness", "citation_quality"]].mean(numeric_only=True))


if __name__ == "__main__":
    main()