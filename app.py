import io, re, regex, numpy as np, pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document

st.set_page_config(page_title="MatchMyResume — AI JD Matcher", page_icon="✨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
:root {
  --accent: #7C3AED;
  --accent-2: #22C55E;
  --bg: #0B1220;
  --panel: #0F172A;
  --muted: #94A3B8;
}
html, body, [class*="css"]  {
  font-family: 'Inter', sans-serif;
}
.block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
.hero {
  background: radial-gradient(90rem 60rem at 10% -10%, rgba(124,58,237,.35), transparent 60%),
              radial-gradient(90rem 60rem at 110% 10%, rgba(34,197,94,.25), transparent 60%);
  border: 1px solid rgba(255,255,255,0.06);
  background-color: var(--panel);
  padding: 1.25rem 1.5rem; border-radius: 18px;
}
.brand { display:flex; align-items:center; gap:.75rem; }
.brand .logo {
  width:38px; height:38px; border-radius:10px;
  background: linear-gradient(135deg, var(--accent), #8B5CF6 50%, #22C55E);
  box-shadow: 0 10px 24px rgba(124,58,237,.25);
}
.tag { display:inline-block; padding:.2rem .55rem; border-radius:999px; font-size:.8rem;
       color:white; background:rgba(124,58,237,.25); border:1px solid rgba(124,58,237,.35) }
.kpi { background: rgba(148,163,184,.08); border:1px solid rgba(255,255,255,.08);
       padding:1rem 1.25rem; border-radius:16px; }
.progress-outer { height:10px; background: rgba(100,116,139,.35); border-radius:8px; overflow:hidden; }
.progress-inner { height:10px; background: linear-gradient(90deg,var(--accent),#8B5CF6,#22C55E); width:0%; }
.input-chip { display:inline-block; margin:.2rem .25rem; padding:.25rem .6rem; border-radius:999px;
              border:1px dashed rgba(148,163,184,.5); color:#e2e8f0; font-size:.85rem; }
.footer { color: var(--muted); font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero">', unsafe_allow_html=True)
colA, colB = st.columns([1, 7])
with colA:
    st.markdown('<div class="brand"><div class="logo"></div><div><h3 style="margin-bottom:.2rem;">MatchMyResume</h3><span class="tag">AI Resume ↔ JD Matcher</span></div></div>', unsafe_allow_html=True)
with colB:
    st.write("Compare any resume with any job description using **Sentence-BERT** embeddings. Get a **semantic fit score**, see **evidence pairs**, and fix **missing skills**.")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("")

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
        return data.decode("utf-8", errors="ignore")

with st.sidebar:
    st.header("⚙️ Controls")
    mode = st.radio("Input mode", ["Upload files", "Paste text"], horizontal=True)

    role_presets = {
        "Data Analyst": "python, sql, tableau, power bi, excel, pandas, scikit-learn, statistics, dashboard, etl, data cleaning",
        "Business Analyst": "requirements gathering, user stories, sql, excel, tableau, stakeholder management, process mapping, a/b testing",
        "BI Analyst": "power bi, tableau, sql, dax, data modeling, star schema, kpis, storytelling, dashboards",
        "Product Analyst": "sql, experiment design, a/b testing, product analytics, mixpanel, amplitude, retention, funnels",
        "Data Scientist": "python, sklearn, pandas, xgboost, statistics, hypothesis testing, feature engineering, model evaluation",
        "ML Engineer": "python, pytorch, tensorflow, scikit-learn, mlflow, docker, kubernetes, deployment, inference, monitoring",
        "Data Engineer": "python, sql, pyspark, spark, airflow, dbt, kafka, snowflake, databricks, aws, azure, docker, ci/cd",
        "NLP Engineer": "nlp, transformers, huggingface, tokenization, embeddings, bert, gpt, rag, vector db, faiss",
        "Computer Vision": "opencv, cnn, pytorch, torchvision, augmentation, detection, segmentation",
        "Financial Analyst": "excel, vba, sql, forecasting, valuation, power bi, tableau, financial modeling",
        "Marketing Analyst": "sql, attribution, mmm, google analytics, a/b testing, segmentation, tableau, power bi",
        "Supply Chain Analyst": "sql, optimization, linear programming, forecasting, inventory, logistics, tableau, power bi",
        "Operations Analyst": "lean six sigma, process improvement, sql, kpis, dashboards, time study",
    }
    role = st.selectbox("Profile (role)", list(role_presets.keys()), index=0)

    domain_presets = {
        "General": "",
        "Healthcare": "hipaa, hl7, readmission, claims, icd, ehr, patient outcomes, compliance",
        "Finance": "risk, kyc, aml, credit scoring, portfolio, trading, fraud detection, regulations",
        "Retail": "demand forecasting, assortment, pricing, market basket, promotions, store ops",
        "Supply Chain": "eta, route optimization, warehouse, picking, lead time, on-time delivery",
        "Sports": "player performance, expected goals, win probability, tracking data, scouting",
        "Education": "student retention, learning analytics, lms, assessment, cohort analysis",
        "Energy": "load forecasting, emissions, grid, renewable, outages, predictive maintenance",
        "Government": "open data, census, public safety, procurement, policy analysis",
        "Cybersecurity": "anomaly detection, SIEM, alerts, threat intel, phishing, zero trust",
    }
    domain = st.selectbox("Domain (industry)", list(domain_presets.keys()), index=0)

    base_keywords = role_presets[role]
    domain_keywords = domain_presets[domain]
    default_keywords = ", ".join([k for k in [base_keywords, domain_keywords] if k])

    kw_text = st.text_area("Keywords (editable, comma-separated)", value=default_keywords, height=110)
    top_k = st.slider("Top evidence pairs", 3, 20, 10)

    st.markdown("---")
    st.caption("Tip: Use text-based PDFs/DOCX (not scans). Paste mode works best for quick checks.")

resume_text = None
jd_text = None

if mode == "Upload files":
    left, right = st.columns(2)
    with left:
        st.subheader("📄 Resume")
        resume_up = st.file_uploader("Upload resume (.pdf/.docx/.txt)", type=["pdf","docx","txt"], key="resume")
    with right:
        st.subheader("🧾 Job Description")
        jd_up = st.file_uploader("Upload JD (.pdf/.docx/.txt)", type=["pdf","docx","txt"], key="jd")
    if resume_up: resume_text = read_any(resume_up)
    if jd_up: jd_text = read_any(jd_up)
else:
    st.subheader("📝 Paste Texts")
    c1, c2 = st.columns(2)
    with c1:
        resume_text = st.text_area("Resume text", height=220, placeholder="Paste resume text here…")
    with c2:
        jd_text = st.text_area("Job description text", height=220, placeholder="Paste JD text here…")
    if st.checkbox("Fill demo samples", value=False):
        resume_text = """
        Data Analyst with Python, SQL, Tableau, and Power BI experience.
        Built ML models in scikit-learn, automated ETL with Airflow, deployed Streamlit apps.
        Cloud: AWS, Azure. Version control with Git. Docker for packaging.
        """
        jd_text = """
        Seeking a Data Analyst proficient in Python and SQL with experience in Tableau or Power BI.
        Familiarity with ETL (Airflow/DBT), Docker, and Git is preferred. Cloud exposure (AWS/Azure) a plus.
        """

go = st.button("🚀 Check Match", type="primary", use_container_width=True)

if go:
    if not resume_text or not jd_text:
        st.error("Please provide both Resume and Job Description (upload or paste).")
        st.stop()

    model = load_model()
    r_txt, j_txt = clean(resume_text), clean(jd_text)

    R, J = embed(model, [r_txt, j_txt])
    overall = cos_sim(R, J)

    jd_sents = split_sentences(j_txt)[:160]
    r_sents  = split_sentences(r_txt)[:500]
    if len(jd_sents) == 0 or len(r_sents) == 0:
        st.warning("One of the texts has no extractable sentences. Try simpler text or DOCX.")
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

    kw_list = [k.strip().lower() for k in kw_text.split(",") if k.strip()]
    r_low = r_txt.lower(); j_low = j_txt.lower()
    have = sorted({k for k in kw_list if k in r_low})
    miss = sorted({k for k in kw_list if (k in j_low and k not in r_low)})

    k1, k2 = st.columns([1, 3])
    with k1:
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.subheader("Fit Score")
        st.write(f"**{overall:.3f}** (cosine)")
        pct = max(min((overall-0.5)/0.5, 1), 0)*100
        st.markdown(f'<div class="progress-outer"><div class="progress-inner" style="width:{pct:.0f}%"></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.subheader("Quick Tips")
        st.write("• Aim for **0.75+** for strong matches")
        st.write("• Add **missing** but truthful keywords from the list below")
        st.write("• Try another **Profile** and **Domain** in the sidebar")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    tab1, tab2, tab3, tab4 = st.tabs(["✅ Skills", "🔍 Evidence", "🧾 Raw Text", "⚙️ Advanced"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Found in Resume")
            if have:
                st.write(" ".join([f'<span class="input-chip">{k}</span>' for k in have]), unsafe_allow_html=True)
            else:
                st.write("—")
        with c2:
            st.subheader("Missing but in JD")
            if miss:
                st.write(" ".join([f'<span class="input-chip">{k}</span>' for k in miss]), unsafe_allow_html=True)
            else:
                st.write("—")

    with tab2:
        st.subheader("Top JD ↔ Resume Matches")
        def _shade(val):
            alpha = max(min((val-0.6)/0.35,1),0)
            return f"background-color: rgba(124,58,237,{alpha:.2f}); color: white" if alpha>0.65 else ""
        styled = df.style.format({"Similarity":"{:.3f}"}).applymap(_shade, subset=["Similarity"])
        st.dataframe(styled, use_container_width=True)

    with tab3:
        st.subheader("Extracted Texts")
        st.text_area("Resume", r_txt, height=220)
        st.text_area("Job Description", j_txt, height=220)

    with tab4:
        st.subheader("Settings Snapshot")
        st.write(f"**Profile:** {role}  |  **Domain:** {domain}")
        st.write("**Keywords used:**")
        st.code(", ".join(kw_list) if kw_list else "(none)")
        st.caption("Tune these in the sidebar for better results.")

st.markdown("---")
st.markdown('<div class="footer">Made with ✨ by you · Theme: violet/emerald · Font: Inter · Tech: Streamlit + Sentence-BERT</div>', unsafe_allow_html=True)
