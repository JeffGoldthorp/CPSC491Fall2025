# Source curation guide

This version assumes the source corpus lives in `sources/`.

## Supported source types

Local files:
- PDF
- DOCX
- TXT
- MD
- HTML / HTM

Website manifests:
- `web_sources.txt`
- `web_sources.json`
- `web_sources.csv`

## Step 1 - Run inventory

```bash
make inventory
```

This creates `data/staged/source_inventory.csv` so you can see the normalized `source_key` for every local file and website.

## Step 2 - Fill in `data/source_register.csv`

For each real source, decide:
- `authority_level`
- `allowed_for_answers`
- `source_type`
- `jurisdiction`
- `freshness_reviewed_on`

## Suggested authority levels

- `primary` - standards, laws, official government guidance, official vendor docs
- `secondary` - respected summaries from recognized bodies
- `local_policy` - county, PSAP, or sponsor-approved SOPs and policies
- `reference` - supporting material, notes, slide decks, or context that should not outrank primary sources

## Suggested starter corpus

- NIST CSF 2.0 and related NIST publications
- CISA Cybersecurity Performance Goals
- CISA resources for 9-1-1 centers
- FCC 911 reliability and outage guidance
- NENA security standards and primers
- APCO guidance
- sponsor-approved local operational documents

## Answer safety rule

Only mark a source as `allowed_for_answers=true` if you are comfortable retrieving and citing that source back to users in the chatbot.
