import json
import argparse
from typing import Dict, Any, List
from app.rag import generate_grounded_response

SAMPLE_QUERIES = [
    "Summarize the main customer complaints and praises",
    "What themes come up about service speed?",
    "Any consistent issues with pricing or value?",
    "Any consistent issues with cleanliness or food safety?",
]

def assert_sms(out: Dict[str, Any]) -> List[str]:
    errors = []
    sms = out.get("sms_draft") or {}
    msgs = sms.get("messages") or []
    if not isinstance(msgs, list):
        return ["sms_draft.messages not a list"]
    for i, m in enumerate(msgs):
        if not isinstance(m, str):
            errors.append(f"sms message {i} not a string")
        elif len(m) > 160:
            errors.append(f"sms message {i} too long ({len(m)} chars)")
    return errors

def assert_grounding_ids(out: Dict[str, Any]) -> List[str]:
    errors = []
    dbg = out.get("debug") or {}
    allowed = set(dbg.get("allowed_chunk_ids") or [])
    if not allowed:
        # If debug not included, we can’t validate this rule.
        return []

    # Collect all referenced ids
    referenced = set()

    for th in out.get("top_themes", []) or []:
        for e in (th.get("evidence") or []):
            cid = e.get("chunk_id")
            if cid:
                referenced.add(cid)

    for it in out.get("recurring_issues", []) or []:
        for cid in (it.get("evidence_chunk_ids") or []):
            referenced.add(cid)

    for it in out.get("isolated_issues", []) or []:
        for cid in (it.get("evidence_chunk_ids") or []):
            referenced.add(cid)

    for it in out.get("ops_recommendations", []) or []:
        for cid in (it.get("grounding_chunk_ids") or []):
            referenced.add(cid)

    bad = sorted([cid for cid in referenced if cid not in allowed])
    if bad:
        errors.append("Found referenced chunk_ids not in allowed_chunk_ids:\n" + "\n".join(bad))
    return errors

def main(top_k: int, min_recurring_reviews: int) -> None:
    all_errors = 0
    for q in SAMPLE_QUERIES:
        out = generate_grounded_response(
            query=q,
            top_k=top_k,
            min_recurring_reviews=min_recurring_reviews,
            include_debug=True,
        )

        errs = []
        errs += assert_sms(out)
        errs += assert_grounding_ids(out)

        if errs:
            all_errors += 1
            print("\n" + "="*80)
            print("QUERY:", q)
            print("FAILED CHECKS:")
            for e in errs:
                print("-", e)
        else:
            print(f"[PASS] {q}")

    if all_errors:
        raise SystemExit(f"\nEvaluation finished with {all_errors} failing queries.")
    print("\nAll checks passed.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=25)
    ap.add_argument("--min_recurring_reviews", type=int, default=2)
    args = ap.parse_args()
    main(args.top_k, args.min_recurring_reviews)
