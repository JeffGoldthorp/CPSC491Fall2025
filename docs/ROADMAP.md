# A-to-Z roadmap for the PSAP 911 cyber risk assistant - source-first version

## Core idea

You already have the real source corpus. That means the build should start from those documents, not from hand-authored placeholder examples.

This version supports exactly that:

1. Real source corpus in `sources/`
2. Source inventory and source register
3. Source-derived supervised fine-tuning candidates
4. Fine-tuning job
5. Source-derived eval candidates
6. Gold eval set review
7. Same corpus ingested into Pinecone for RAG
8. Base vs finetuned vs rag vs hybrid comparison

## Checklist

### Phase 0 - Source setup
- [x] Support a top-level `sources/` directory
- [x] Support PDFs, DOCX, TXT, MD, HTML, and HTM
- [x] Support website manifests via text, JSON, and CSV
- [x] Add source inventory builder
- [ ] Populate `sources/` with the real corpus
- [ ] Review `source_inventory.csv`
- [ ] Finalize `data/source_register.csv`

### Phase 1 - Fine-tuning from real sources
- [x] Add source-derived SFT dataset builder
- [x] Keep optional manual example path available
- [x] Validate generated JSONL files
- [ ] Generate first candidate set
- [ ] Review candidate quality
- [ ] Launch the first fine-tuning job
- [ ] Save model id in `.env`

### Phase 2 - Evals
- [x] Add eval bootstrap generator from the same source corpus
- [x] Keep reviewed gold eval set path
- [x] Keep single-mode and compare-all eval runners
- [ ] Review and promote eval candidates into `gold_eval_set.jsonl`
- [ ] Evaluate base vs finetuned

### Phase 3 - Retrieval
- [x] Reuse `sources/` for RAG ingestion
- [x] Reuse `data/source_register.csv` for answer filtering
- [x] Upsert chunks and metadata into Pinecone
- [ ] Tune chunk size and top-k
- [ ] Add reranking later if needed

### Phase 4 - Hybrid demo
- [x] Keep base, finetuned, rag, and hybrid modes
- [x] Keep Streamlit demo
- [x] Keep FastAPI backend
- [ ] Add auth and audit logging for production
- [ ] Add deployment config

## Recommended working order for your team

### Step A - Put everything in `sources/`
- PDFs
- Word docs
- policy notes
- link manifests for websites

### Step B - Run inventory
Produce `data/staged/source_inventory.csv` and classify the sources.

### Step C - Generate SFT candidates
Use the real source corpus to build the first source-derived train and validation sets.

### Step D - Fine-tune and test
Create the fine-tuned model and test its behavior.

### Step E - Build the eval set
Use the source-derived eval bootstrap file as raw material, then hand-review the best questions.

### Step F - Add RAG
Ingest the same source corpus into Pinecone and evaluate RAG and hybrid.

### Step G - Demo and iterate
Use Streamlit for the class demo. Move to Next.js only if you want a more product-like UI later.

## Best practice reminders

- Fine-tuning is for behavior, style, vocabulary, and consistency.
- Retrieval is for factual grounding, citations, and changeable knowledge.
- The source register matters a lot. Poor source metadata leads to poor retrieval quality.
- Auto-generated SFT and eval candidates are useful accelerators, but the strongest results come from reviewing them before relying on them.
