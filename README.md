# AI Resume ↔ JD Matcher (Streamlit, One-Click Deploy)

Public web app that compares a resume with a job description using **Sentence-BERT embeddings** and shows:
- An **overall semantic fit score** (0–1)
- **Evidence sentences** (JD lines and the best-matching resume lines)
- **Keyword gap analysis** (found vs missing skills)

## 🧪 Try locally

```bash
pip install -r requirements.txt
streamlit run app.py
# Opens at http://localhost:8501
```

## ☁️ Deploy to Streamlit Community Cloud (Free)

1. Push these files to a **public GitHub repo**.
2. Go to https://share.streamlit.io/ → **New app** → Select your repo.
3. **Main file**: `app.py`
4. Deploy. Share your public URL!

## 🚀 Deploy to Hugging Face Spaces (Free)

1. Create a new Space → **Streamlit** template.
2. Upload `app.py` and `requirements.txt` (or connect your GitHub repo).
3. Space builds automatically and gives you a public link.

## 📁 Project Structure

```
.
├── app.py              # Streamlit app (frontend + inference)
├── requirements.txt    # Python deps
└── README.md
```

## 📝 Notes
- The first run downloads the model (`all-MiniLM-L6-v2`) which may take a minute.
- Use text-based PDFs or DOCX for reliable extraction. Scanned PDFs (images) won’t extract text without OCR.
- Edit the keyword list in the sidebar to match your target role (e.g., Data Analyst, Data Engineer).

## 🧭 Roadmap
- Add OCR for scanned PDFs (tesseract/pytesseract)
- Add export to PDF/CSV
- Add tokenizer-based highlighting for matched phrases