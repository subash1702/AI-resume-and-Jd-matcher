# MatchMyResume — AI Resume ↔ JD Matcher (Trendy UI)

A stylish Streamlit app that compares any resume to any job description using **Sentence-BERT** embeddings.
It shows a **semantic fit score**, **top evidence pairs**, and **keyword gaps** with modern dark theme,
role presets (profiles), and domain (industry) presets.

## ✨ Features
- Upload **or paste** texts
- **Profiles**: Data/BI/ML/DE/Product/BA/Finance/Marketing/etc.
- **Domains**: Healthcare, Finance, Retail, Supply Chain, Sports, Education, Energy, Government, Cybersecurity, General
- Trendy **violet/emerald** gradient + **Inter** font
- Tabs: Skills, Evidence, Raw Text, Advanced

## ▶️ Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501

## ☁️ Deploy
- **Streamlit Community Cloud**: New app → main file: `app.py`
- **Hugging Face Spaces** (Streamlit template)

> First run downloads the model `sentence-transformers/all-MiniLM-L6-v2`.

## Notes
- Works best with text-based PDFs or DOCX (not scanned images)
- You can customize keyword presets in the sidebar or edit `app.py`
