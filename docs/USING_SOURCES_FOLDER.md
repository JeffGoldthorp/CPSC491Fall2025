# Using the `sources/` folder

Put your real project sources in the top-level `sources/` folder.

Supported local files:
- PDF
- DOCX
- TXT
- MD
- HTML / HTM

For websites, create one of these files inside `sources/`:
- `web_sources.txt`
- `web_sources.json`
- `web_sources.csv`

Example `web_sources.txt`:

```text
https://www.cisa.gov/resources-tools/resources/cybersecurity-resources-9-1-1-centers
https://www.nist.gov/cyberframework
```

Notes:
- one URL per line for `.txt`
- for `.json`, use either a list of URLs or `{ "urls": [...] }`
- for `.csv`, add a `url` column
