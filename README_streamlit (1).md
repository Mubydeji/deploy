# Streamlit deployment notes

This version uses **no sidebar setup UI**.

## What to do

1. Add your API key to **Streamlit Secrets**:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

2. Keep your JSONL file in the repo backend, either as:

- `nfti_pages.jsonl`
- `data/nfti_pages.jsonl`

Recommended repo structure:

```text
.
├── app.py
├── requirements.txt
├── nfti_pages.jsonl
└── README_streamlit.md
```

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

- Push `app.py`, `requirements.txt`, and `nfti_pages.jsonl` to GitHub.
- In Streamlit app settings, add `GEMINI_API_KEY` under **Secrets**.
- Redeploy.

## Notes

- The sidebar is collapsed by default.
- The app will stop with a clear error if the secret or JSONL backend file is missing.
- The app reads only from backend files now; there is no file uploader.
