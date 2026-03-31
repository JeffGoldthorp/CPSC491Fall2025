from __future__ import annotations

import argparse
import subprocess
from pathlib import Path



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-dir", default="data/staged/evals")
    parser.add_argument("--namespace", default="public_authoritative")
    parser.add_argument("--ft-model")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ("base", None),
        ("finetuned", args.ft_model),
        ("rag", None),
        ("hybrid", args.ft_model),
    ]

    for mode, model_override in runs:
        cmd = [
            "python",
            "-m",
            "app.evals.run_eval",
            "--eval-file",
            args.eval_file,
            "--mode",
            mode,
            "--namespace",
            args.namespace,
            "--output-csv",
            str(output_dir / f"{mode}.csv"),
        ]
        if model_override:
            cmd.extend(["--model-override", model_override])
        subprocess.run(cmd, check=True)
        print(f"Finished {mode} eval")


if __name__ == "__main__":
    main()
