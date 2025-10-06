import io, re, regex, numpy as np, pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document

# -------------------- Page & Theme --------------------
st.set_page_config(
    page_title="AI Resume ↔ JD Matcher",
    page_icon="🔎",
    layout="wide"
)

# Small CSS polish
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 3rem;}
.kpi-card {background: var(--secondary-background-color); padding: 1rem 1.25rem; border-radius: 16px; border: 1px solid rgba(0,0,0,0.05);}
.footer { color: #6b7280; font-size: 0.9rem; }
hr { margin: 0.8rem 0; }
code { background: rgba(0,0,0,.04); padding: .15rem .35rem; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# -------------------- Header --------------------
col_logo, col_title = st.columns([1,6])
with col_logo:
    st.image("https://raw.githubusercontent.com/streamlit/brand/master/logomark/streamlit-mark-logo-primary-colormark-darktext.png", width=56)
with col_title:
    st.title("AI Resume ↔ JD Matcher")
    st.caption("Assess resume–job fit with semantic embeddings (Sentence‑BERT), evidence pairs, and keyword gap analysis.")

st.divider()

# -------------------- Helpers --------------------
@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def clean(text: str) -> str:
    text = text.replace("\u00A0"," ").strip()
    text = regex.sub(r"\s+", " ", text)
    return text

def split_sentences(text: str):
    return [t.strip() for t in re.split(r'(?<=[.!?])\s+', text) if t.strip()]

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
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        # assume text
        return data.decode("utf-8", errors="ignore")

# -------------------- Sidebar (Controls) --------------------
with st.sidebar:
    st.header("🛠️ Settings")

    presets = {
        "Data Analyst": "python, sql, tableau, power bi, excel, pandas, scikit-learn, statistics, dashboard, etl, data cleaning",
        "Data Engineer": "python, sql, pyspark, spark, airflow, dbt, snowflake, databricks, aws, azure, kafka, docker, ci/cd",
        "ML Engineer": "python, pytorch, tensorflow, scikit-learn, mlflow, docker, kubernetes, sagemaker, feature store, inference, monitoring"
    }
    role = st.selectbox("Role preset", list(presets.keys()), index=0)
    default_keywords = presets[role]

    kw_text = st.text_area("Keywords (comma-separated)", value=default_keywords, height=90, help="Used for the keyword gap check")

    top_k = st.slider("Top evidence pairs", 3, 20, 10, help="How many JD↔Resume pairs to show")
    st.markdown("---")
    place_demo = st.checkbox("Use built‑in demo texts", value=False)
    st.caption("Tip: upload text-based PDFs/DOCX (not scans).")

# -------------------- Inputs --------------------
left, right = st.columns(2)
with left:
    st.subheader("📄 Resume")
    resume_up = st.file_uploader("Upload resume (.pdf/.docx/.txt)", type=["pdf","docx","txt"], key="resume")
with right:
    st.subheader("🧾 Job Description")
    jd_up = st.file_uploader("Upload JD (.pdf/.docx/.txt)", type=["pdf","docx","txt"], key="jd")

# Demo fallback
resume_text = None
jd_text = None
if place_demo:
    resume_text = """
    Data Analyst with Python, SQL, Tableau, and Power BI experience.
    Built ML models in scikit-learn, automated ETL with Airflow, deployed Streamlit apps.
    Cloud: AWS, Azure. Version control with Git. Docker for packaging.
    """
    jd_text = """
    Seeking a Data Analyst proficient in Python and SQL with experience in Tableau or Power BI.
    Familiarity with ETL (Airflow/DBT), Docker, and Git is preferred. Cloud exposure (AWS/Azure) a plus.
    """
else:
    if resume_up: resume_text = read_any(resume_up)
    if jd_up: jd_text = read_any(jd_up)

st.markdown("")
go = st.button("🚀 Compute Match", type="primary", use_container_width=True)

# -------------------- Inference --------------------
if go:
    if not resume_text or not jd_text:
        st.error("Please upload both files or enable demo texts in the sidebar.")
        st.stop()

    model = load_model()
    r_txt, j_txt = clean(resume_text), clean(jd_text)

    # Overall similarity
    R, J = embed(model, [r_txt, j_txt])
    overall = cos_sim(R, J)

    # Sentences & embeddings
    jd_sents = split_sentences(j_txt)[:120]
    r_sents  = split_sentences(r_txt)[:400]

    if len(jd_sents) == 0 or len(r_sents) == 0:
        st.warning("One of the files appears to have no extractable text. Try DOCX or TXT, or a non-scanned PDF.")
        st.stop()

    E_jd = embed(model, jd_sents)
    E_r  = embed(model, r_sents)
    sims = np.matmul(E_jd, E_r.T)

    pairs = []
    for i, row in enumerate(sims):
        j_best = int(np.argmax(row))
        pairs.append((jd_sents[i], r_sents[j_best], float(row[j_best])))
    pairs = sorted(pairs, key=lambda x: -x[2])[:top_k]
    df = pd.DataFrame(pairs, columns=["JD Sentence", "Resume Sentence", "Similarity"])

    # Keyword gap
    kw_list = [k.strip().lower() for k in kw_text.split(",") if k.strip()]
    r_low = r_txt.lower(); j_low = j_txt.lower()
    have = sorted({k for k in kw_list if k in r_low})
    miss = sorted({k for k in kw_list if (k in j_low and k not in r_low)})

    # -------------- KPIs --------------
    k1, k2 = st.columns([1,3])
    with k1:
        with st.container():
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.metric("Overall Semantic Fit", f"{overall:.3f}")
            st.progress(min(max((overall - 0.5) / 0.5, 0.0), 1.0))
            st.markdown('</div>', unsafe_allow_html=True)
    with k2:
        with st.container():
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.write("**Quick Tips**")
            st.write("• Aim for 0.75+ for close fits\n• Add relevant missing keywords only if true\n• Tweak the role preset in the sidebar")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    tab1, tab2, tab3 = st.tabs(["✅ Skills", "🔍 Evidence", "🧾 Raw Text"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Found in Resume")
            if have: st.success(", ".join(have))
            else: st.write("—")
        with c2:
            st.subheader("Missing but in JD")
            if miss: st.error(", ".join(miss))
            else: st.write("—")

    with tab2:
        st.subheader("Top JD ↔ Resume Matches")
        def _shade(val):
            # green intensity by similarity
            return f"background-color: rgba(16,185,129,{max(min((val-0.6)/0.35,1),0):.2f})"
        styled = df.style.format({"Similarity":"{:.3f}"}).applymap(_shade, subset=["Similarity"])
        st.dataframe(styled, use_container_width=True)

    with tab3:
        st.subheader("Extracted Texts")
        st.text_area("Resume", r_txt, height=220)
        st.text_area("Job Description", j_txt, height=220)

st.markdown("---")
st.caption('Built with Streamlit + Sentence‑BERT (`all-MiniLM-L6-v2`). · Add your links: GitHub/LinkedIn in the footer below.')
st.markdown('<div class="footer">Made by <a href="https://github.com/" target="_blank">you</a> · Consider starring the repo ⭐</div>', unsafe_allow_html=True)