# PSAP 911 Cyber Risk Assistant - source-first classroom workflow

This version of the project is designed around your real workflow:

1. Put your real source PDFs, Word docs, text files, and website links in `sources/`
2. Auto-inventory those sources
3. Generate source-derived fine-tuning examples from the corpus
4. Fine-tune the model
5. Generate and review eval candidates
6. Ingest the same source corpus into Pinecone
7. Compare base vs finetuned vs RAG vs hybrid
8. Demo in Streamlit

## Folder layout

```text
.
├── app/
│   ├── api/
│   ├── evals/
│   ├── ingest/
│   ├── services/
│   ├── training/
│   └── ui/
├── data/
│   ├── approved_examples/        # optional manual examples
│   ├── evals/                    # reviewed gold eval set lives here
│   ├── staged/                   # generated outputs
│   └── source_register.csv       # source authority and policy metadata
├── docs/
├── frontend-nextjs/
└── sources/                      # your real PDFs, docx, txt, html, and web link manifests
```

## What belongs in `sources/`

Supported local files:
- `.pdf`
- `.docx`
- `.txt`
- `.md`
- `.html`
- `.htm`

Supported website manifests:
- `web_sources.txt` with one URL per line
- `web_sources.json` as a list of URLs or `{ "urls": [...] }`
- `web_sources.csv` with a `url` column

Example:

```text
sources/
├── NIST_CSF_2_0_Crosswalk_NENA.pdf
├── NG911 Cybersecurity Primer.pdf
├── local_policy.docx
└── web_sources.txt
```

Example `web_sources.txt`:

```text
https://www.cisa.gov/resources-tools/resources/cybersecurity-resources-9-1-1-centers
https://www.nist.gov/cyberframework
```

## Quick start

### 1. Create and activate a virtual environment

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Create `.env`

Copy `.env.example` to `.env` and fill in:

```env
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-5-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=1536
OPENAI_FINE_TUNE_BASE_MODEL=gpt-4o-mini
OPENAI_FINE_TUNE_SUFFIX=psap-911-cyber-sft
OPENAI_FINETUNED_MODEL=

PINECONE_API_KEY=
PINECONE_INDEX_HOST=
PINECONE_NAMESPACE=public_authoritative
PINECONE_TOP_K=8
```

### 3. Inventory the sources

```bash
make inventory
```

This writes `data/staged/source_inventory.csv`.

### 4. Update `data/source_register.csv`

Use the inventory file to mark each source with:
- `authority_level`
- `allowed_for_answers`
- `source_type`
- `jurisdiction`
- `freshness_reviewed_on`

### 5. Generate source-derived fine-tuning data

```bash
make sft-dataset
```

Outputs:
- `data/staged/sft_candidates.jsonl`
- `data/staged/sft_train.jsonl`
- `data/staged/sft_val.jsonl`

### 6. Launch the fine-tuning job

```bash
make ft-job
```

When the job finishes, save the returned model id into `.env`:

```env
OPENAI_FINETUNED_MODEL=ft:...
```

### 7. Test the fine-tuned model

```bash
make ft-chat
```

### 8. Bootstrap eval questions from the same sources

```bash
make eval-bootstrap
```

This writes `data/staged/eval_candidates.jsonl`. Review the best items and copy them into `data/evals/gold_eval_set.jsonl`.

### 9. Evaluate the fine-tuned model

```bash
make eval-ft
```

### 10. Ingest the corpus for RAG

```bash
make ingest
```

### 11. Run comparison evals

```bash
make eval-rag
make eval-hybrid
make eval-all
```

### 12. Run the API and Streamlit UI

```bash
make api
make ui
```

## Important notes

- Fine-tuning still needs chat-style JSONL, so the project now creates that automatically from your source corpus.
- The generated SFT and eval candidates are a fast bootstrap, not a substitute for human review.
- `data/approved_examples/` is still available if you want to hand-author or sponsor-review a smaller, higher-quality set later.
- `sources/` is the canonical folder for your real source material in this version.
