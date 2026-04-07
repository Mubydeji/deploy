# Streamlit deployment files for NFTI Knowledge Assistant

## Files
- `app.py` — Streamlit version of your notebook
- `requirements.txt` — Python dependencies

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud
1. Create a GitHub repo and upload:
   - `app.py`
   - `requirements.txt`
   - optionally `nfti_pages.jsonl`
2. In Streamlit app settings, add a secret:
   - `GEMINI_API_KEY = "your_key_here"`
3. Deploy the repo.

## Important note
If you do not commit `nfti_pages.jsonl`, the app still works because it lets the user upload that file from the sidebar.
