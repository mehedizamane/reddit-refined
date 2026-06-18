# Reddit Refined: Streamlining Data Pre-Processing for Reddit Data

A pipeline built to process raw Reddit `.jsonl` data logs into clean, query-ready `.parquet` formats.

## Key Features
- Seamlessly processes both Submissions (Posts) and Comments using an adaptive single-pass parser model.
- Strips markdown tags, raw hyperlink texts, and breaks down messy HTML encoding parameters.
- Extracts raw graphical emojis and markdown quotes into explicit metadata metric tracking vectors.
- Tags structural depth layout logic (`is_root_reply`) by validating Reddit database prefix hooks (`t1_` vs `t3_`).
- Computes lexical complexity scores and tracks conversational delay intervals (`response_latency_mins`).

## 🛠️ Installation
Install the necessary processing dependencies through your system console terminal:
```bash
pip install pandas pyarrow
