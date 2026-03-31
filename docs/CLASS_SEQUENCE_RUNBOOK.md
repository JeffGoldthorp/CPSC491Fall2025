# Classroom-sequence runbook - using your real sources folder

This version assumes your real documents already live in `sources/`.

## The exact order

1. Inventory sources
2. Classify them in `data/source_register.csv`
3. Generate source-derived SFT examples
4. Fine-tune the model
5. Test the fine-tuned model
6. Bootstrap eval questions and build the reviewed gold eval set
7. Evaluate the fine-tuned model
8. Ingest the same source corpus into Pinecone
9. Evaluate RAG and hybrid
10. Demo the system

## Commands

### 1) Inventory everything under `sources/`

```bash
make inventory
```

### 2) Review `data/staged/source_inventory.csv` and update `data/source_register.csv`

At minimum, set these columns correctly:
- `source_key`
- `authority_level`
- `allowed_for_answers`
- `source_type`
- `jurisdiction`
- `freshness_reviewed_on`

### 3) Generate source-derived SFT datasets

```bash
make sft-dataset
```

### 4) Launch the fine-tuning job

```bash
make ft-job
```

### 5) Save the model id in `.env`

```env
OPENAI_FINETUNED_MODEL=ft:your-model-id-here
```

### 6) Test the fine-tuned model only

```bash
make ft-chat
```

### 7) Bootstrap eval questions

```bash
make eval-bootstrap
```

Now review `data/staged/eval_candidates.jsonl` and copy the best items into `data/evals/gold_eval_set.jsonl`.

### 8) Evaluate the fine-tuned model

```bash
make eval-ft
```

### 9) Ingest the same corpus for RAG

```bash
make ingest
```

### 10) Evaluate retrieval and hybrid

```bash
make eval-rag
make eval-hybrid
make eval-all
```

### 11) Launch the demo app

```bash
make api
make ui
```

## What is different now

Old workflow:
- hand-write examples first
- later add documents

New workflow:
- put your real PDFs and websites in `sources/`
- generate SFT examples from the source corpus
- use the same corpus later for RAG

## What to watch for

### Fine-tuned only
Good at PSAP tone, structure, scope control, and repeated guidance patterns.

### RAG only
Good at grounding and citations, especially for standards, reports, and evolving guidance.

### Hybrid
Usually best once you have a good source register and a decent fine-tuned model.
