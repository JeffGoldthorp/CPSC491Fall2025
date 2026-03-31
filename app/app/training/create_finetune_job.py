from __future__ import annotations

import argparse
import time
from pathlib import Path

from app.clients import get_openai_client
from app.config import settings



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-file", required=True)
    parser.add_argument("--validation-file")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--suffix")
    args = parser.parse_args()

    client = get_openai_client()

    with Path(args.training_file).open("rb") as f:
        training = client.files.create(file=f, purpose="fine-tune")

    validation_id = None
    if args.validation_file:
        with Path(args.validation_file).open("rb") as f:
            validation = client.files.create(file=f, purpose="fine-tune")
            validation_id = validation.id

    job = client.fine_tuning.jobs.create(
        training_file=training.id,
        validation_file=validation_id,
        model=settings.openai_fine_tune_base_model,
        suffix=args.suffix or settings.openai_fine_tune_suffix,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "n_epochs": "auto",
                    "batch_size": "auto",
                    "learning_rate_multiplier": "auto",
                }
            },
        },
    )
    print(f"Created fine-tuning job: {job.id}")

    if args.wait:
        while True:
            current = client.fine_tuning.jobs.retrieve(job.id)
            print(f"status={current.status}")
            if current.status in {"succeeded", "failed", "cancelled"}:
                print(current)
                break
            time.sleep(30)


if __name__ == "__main__":
    main()
