# AI Resume ↔ JD Matcher (Aesthetic UI)

A polished Streamlit app that compares a resume with a job description using **Sentence‑BERT embeddings**.  
Shows **overall semantic fit**, **top evidence pairs**, and **keyword gaps** with a clean, modern UI.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
- **Streamlit Community Cloud**: New app → point to `app.py`
- **Hugging Face Spaces**: Create Space (Streamlit) → upload files

## Notes
- First run downloads `all-MiniLM-L6-v2` (may take ~1–2 min)
- Works best with text-based PDFs or DOCX (not scanned images)