import streamlit as st
import numpy as np
import re, regex
import pandas as pd

from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document

st.set_page_config(page_title="AI Resume ↔ JD Matcher", layout="wide")

st.title("🔎 AI Resume ↔ Job Description Matcher")
st.caption("Upload a resume and a job description (PDF/DOCX/TXT). The app computes a semantic fit score using Sentence-BERT and shows evidence lines.")

@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def clean(text: str) -> str:
    text = text.replace("\\u00A0", " ").strip()
    text = regex.sub(r"\\s+", " ", text)
    return text

def split_sentences(text: str):
    return [t.strip() for t in re.split(r'(?<=[.!?])\\s+', text) if t.strip()]

def embed(model, texts):
    return model.encode(texts, normalize_embeddings=True)

def cos_sim(a, b) -> float:
    return float(np.dot(a, b))

def read_any(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        pdf = PdfReader(io.BytesIO(data))
        return " ".join([p.extract_text() or "" for p in pdf.pages])
    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\\n".join([p.text for p in doc.paragraphs])
    else:
        # assume text
        return data.decode("utf-8", errors="ignore")

import io

with st.sidebar:
    st.header("Upload Files")
    resume = st.file_uploader("Resume (.pdf/.docx/.txt)", type=["pdf","docx","txt"], key="resume")
    jd = st.file_uploader("Job Description (.pdf/.docx/.txt)", type=["pdf","docx","txt"], key="jd")
    top_k = st.slider("Top evidence pairs", min_value=3, max_value=20, value=10, step=1)
    st.markdown("---")
    st.markdown("**Keyword Pack (edit as needed):**")
    default_keywords = "python, sql, tableau, power bi, excel, scikit-learn, pandas, numpy, docker, git, airflow, dbt, pyspark, snowflake, aws, azure, gcp, mlflow, fastapi, streamlit"
    kw_text = st.text_area("Keywords (comma-separated)", value=default_keywords, height=100)

place_demo = st.checkbox("Use tiny demo texts", value=False, help="If you don't have files yet, tick this to use sample texts.")

if place_demo:
    resume_text = \"\"\"
    Data Analyst with Python, SQL, Tableau, and Power BI experience.
    Built ML models in scikit-learn, automated ETL with Airflow, deployed Streamlit apps.
    Cloud: AWS, Azure. Version control with Git. Docker for packaging.
    \"\"\"
    jd_text = \"\"\"
    Seeking a Data Analyst proficient in Python and SQL with experience in Tableau or Power BI.
    Familiarity with ETL (Airflow/DBT), Docker, and Git is preferred. Cloud exposure (AWS/Azure) a plus.
    \"\"\"
else:
    resume_text = None
    jd_text = None
    if resume is not None:
        resume_text = read_any(resume)
    if jd is not None:
        jd_text = read_any(jd)

model = load_model()

if st.button("Compute Match", type="primary"):
    if not resume_text or not jd_text:
        st.error("Please upload both files or enable demo texts.")
    else:
        r_txt, j_txt = clean(resume_text), clean(jd_text)
        R, J = embed(model, [r_txt, j_txt])
        overall = cos_sim(R, J)

        jd_sents = split_sentences(j_txt)[:120]
        r_sents  = split_sentences(r_txt)[:400]

        if len(jd_sents) == 0 or len(r_sents) == 0:
            st.warning("One of the files has no extractable text. Try a different format (TXT/DOCX) or a non-scanned PDF.")
        else:
            E_jd = embed(model, jd_sents)
            E_r  = embed(model, r_sents)

            sims = np.matmul(E_jd, E_r.T)  # cosine due to normalized embeddings
            pairs = []
            for i, row in enumerate(sims):
                j_best = int(np.argmax(row))
                pairs.append((jd_sents[i], r_sents[j_best], float(row[j_best])))
            pairs = sorted(pairs, key=lambda x: -x[2])[:top_k]

            st.metric("Overall Semantic Fit (0–1)", f"{overall:.3f}")
            st.progress(min(max((overall - 0.5) / 0.5, 0.0), 1.0))

            # Keyword gap
            kw_list = [k.strip().lower() for k in kw_text.split(",") if k.strip()]
            r_low = r_txt.lower()
            j_low = j_txt.lower()
            have = sorted({k for k in kw_list if k in r_low})
            miss = sorted({k for k in kw_list if (k in j_low and k not in r_low)})

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("✅ Found in Resume")
                st.write(", ".join(have) if have else "—")
            with c2:
                st.subheader("❌ Missing but in JD")
                st.write(", ".join(miss) if miss else "—")

            st.subheader("🔍 Evidence (JD requirement → Resume line)")
            df = pd.DataFrame(pairs, columns=["JD Sentence", "Resume Sentence", "Similarity"])
            st.dataframe(df, use_container_width=True)

            with st.expander("Show raw extracted texts"):
                st.text_area("Resume Text", r_txt, height=200)
                st.text_area("JD Text", j_txt, height=200)

st.markdown("---")
st.caption("Built with Streamlit + Sentence-BERT (all-MiniLM-L6-v2). For best results, upload text-based PDFs or DOCX (not scanned images).")