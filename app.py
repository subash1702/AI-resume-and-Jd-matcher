import io, re, regex, numpy as np, pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document

# ---------------- Page config ----------------
st.set_page_config(page_title="MatchMyResume — AI JD Matcher", page_icon="✍️", layout="wide")

# Optional: set TITLE_URL from Secrets; user can override in sidebar
DEFAULT_TITLE_URL = st.secrets.get("TITLE_URL", "https://github.com/yourname/yourrepo")

# ---------------- Fonts & CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;700&family=Inter:wght@400;600;800&display=swap');

:root{
  --violet:#7C3AED;
  --violet2:#8B5CF6;
  --emerald:#22C55E;
  --bg:#0B1220;
  --panel:#0F172A;
  --muted:#94A3B8;
}

html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
.block-container{max-width:1080px;padding-top:1rem;padding-bottom:2rem;}

/* HERO */
.hero{position:relative;background:#0F172A;border:1px solid rgba(255,255,255,.06);
      border-radius:20px;padding:18px 18px 16px 18px;
      box-shadow: 0 10px 26px rgba(0,0,0,.25);
      overflow:hidden;}
.hero:before{
  content:'';position:absolute;inset:-40% -30% auto auto;height:340px;width:340px;
  background:radial-gradient(closest-side, rgba(124,58,237,.28), transparent 70%);
  filter: blur(2px);
}
.hero:after{
  content:'';position:absolute;inset:auto auto -35% -30%;height:320px;width:320px;
  background:radial-gradient(closest-side, rgba(34,197,94,.22), transparent 70%);
  filter: blur(2px);
}

/* Brand row */
.brand{display:flex;align-items:center;gap:14px;z-index:2;position:relative;}
.brand-badge{width:44px;height:44px;border-radius:12px;
  background:linear-gradient(135deg,var(--violet),var(--violet2) 55%,var(--emerald));
  box-shadow:0 10px 26px rgba(124,58,237,.35);}

/* TYPEWRITER TITLE */
.brand-title{font-family:'IBM Plex Mono', monospace; font-weight:700; letter-spacing: .5px;
  font-size: clamp(30px, 4.2vw, 40px); color:#F8FAFC; line-height:1.06; margin:0; white-space:nowrap;}
.brand-sub{margin:.25rem 0 0 0; color:#E2E8F0;}

/* Small pill */
.pill{display:inline-block;margin-top:6px;padding:.22rem .6rem;border-radius:999px;
  font-size:.78rem;color:#fff;background:rgba(124,58,237,.28);border:1px solid rgba(124,58,237,.38)}

/* Cards & KPI */
.card{background:rgba(148,163,184,.08);border:1px solid rgba(255,255,255,.08);
      border-radius:16px;padding:1rem 1.25rem;}
.kpi-big{font-size:1.3rem;font-weight:800;}
.progress-outer{height:10px;background:rgba(100,116,139,.35);border-radius:8px;overflow:hidden;}
.progress-inner{height:10px;background:linear-gradient(90deg,var(--violet),var(--violet2),var(--emerald));width:0%;}

/* Inputs */
.chips span{display:inline-block;margin:.22rem .28rem;padding:.28rem .65rem;border-radius:999px;
  border:1px solid rgba(148,163,184,.4);background:#1F2937;color:#e2e8f0;font-size:.85rem;}

/* Buttons */
.stButton>button{background:linear-gradient(90deg,var(--violet),var(--violet2),var(--emerald));
  color:#fff;border:none;border-radius:12px;padding:.7rem 1.1rem;font-weight:800;}
.stButton>button:hover{filter:brightness(1.06);}

/* Footer */
.footer{color:var(--muted);font-size:.92rem;margin-top:2rem;}
</style>
""", unsafe_allow_html=True)

# ---------------- Header (clickable title) ----------------
def hero(title_url: str):
    st.markdown(f"""
    <div class="hero">
      <div class="brand">
        <div class="brand-badge"></div>
        <div>
          <a href="{title_url}" target="_blank" style="text-decoration:none;">
            <div class="brand-title">MatchMyResume</div>
          </a>
          <div class="brand-sub">AI Resume ↔ JD Matcher · Sentence-BERT embeddings</div>
          <div class="pill">Type, paste or upload · Get fit score + evidence · Spot missing skills</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Helpers ----------------
@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def clean(t): return regex.sub(r"\s+", " ", t.replace("\u00A0"," ").strip())
def split_sentences(t): return [s.strip() for s in re.split(r'(?<=[.!?])\s+', t) if s.strip()]
def embed(m, texts): return m.encode(texts, normalize_embeddings=True)
def cos_sim(a,b): return float(np.dot(a,b))

def read_any(uploaded_file)->str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        pdf = PdfReader(io.BytesIO(data))
        return " ".join([p.extract_text() or "" for p in pdf.pages])
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join([p.text for p in doc.paragraphs])
    return data.decode("utf-8", errors="ignore")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Controls")
    title_url = st.text_input("Title link (optional)", value=DEFAULT_TITLE_URL, help="Users can click the big title to open this link.")
    role_presets = {
        "Data Analyst":"python, sql, tableau, power bi, excel, pandas, scikit-learn, statistics, dashboard, etl, data cleaning",
        "Business Analyst":"requirements gathering, user stories, sql, excel, stakeholder management, process mapping, a/b testing",
        "Data Engineer":"python, sql, pyspark, spark, airflow, dbt, kafka, snowflake, databricks, aws, azure, docker, ci/cd",
        "ML Engineer":"python, pytorch, tensorflow, scikit-learn, mlflow, docker, kubernetes, deployment, monitoring",
        "BI Analyst":"power bi, tableau, sql, dax, data modeling, storytelling, dashboards",
        "Product Analyst":"sql, a/b testing, experiment design, product analytics, retention, funnels",
        "Financial Analyst":"excel, vba, sql, forecasting, valuation, power bi, tableau",
        "Marketing Analyst":"sql, attribution, google analytics, segmentation, tableau, power bi",
        "Supply Chain Analyst":"sql, optimization, forecasting, inventory, logistics, tableau, power bi",
        "Operations Analyst":"lean six sigma, process improvement, sql, kpis, dashboards",
        "General":"communication, stakeholder management, reporting, presentation"
    }
    domain_presets = {
        "General":"",
        "Healthcare":"hipaa, hl7, claims, ehr, icd, patient outcomes",
        "Finance":"risk, kyc, aml, credit scoring, portfolio, fraud detection",
        "Retail":"pricing, promotions, assortment, demand forecasting, basket analysis",
        "Supply Chain":"route optimization, warehouse, delivery, inventory",
        "Education":"student retention, lms, assessment, analytics",
        "Energy":"load forecasting, renewable, grid, maintenance",
        "Cybersecurity":"siem, detection, alerts, threat intel, phishing",
    }
    role = st.selectbox("Profile (role)", list(role_presets.keys()), index=0)
    domain = st.selectbox("Domain (industry)", list(domain_presets.keys()), index=0)
    kw_text = st.text_area("Keywords (editable, comma-separated)",
        value=", ".join([x for x in [role_presets[role], domain_presets[domain]] if x]), height=110)
    top_k = st.slider("Top evidence pairs", 3, 20, 10)

# Render hero after reading sidebar link
hero(title_url)

# ---------------- Unified Inputs ----------------
st.subheader("Input (paste takes priority over upload)")

def get_text_source(title, upload_help, up_key, ta_key):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{title} — Upload**")
        up = st.file_uploader(upload_help, type=["pdf","docx","txt"], key=up_key)
    with c2:
        st.markdown(f"**{title} — Paste**")
        txt = st.text_area(f"{title} text", height=220, key=ta_key, placeholder=f"Paste {title.lower()} text here…")
    if txt and txt.strip(): return txt.strip()
    if up: return read_any(up)
    return None

resume_text = get_text_source("Resume", "Upload resume (.pdf/.docx/.txt)", "resume_up", "resume_ta")
jd_text = get_text_source("Job Description", "Upload JD (.pdf/.docx/.txt)", "jd_up", "jd_ta")

if st.checkbox("Fill demo samples (if empty)", value=False):
    if not resume_text: resume_text = "Data Analyst with Python, SQL, Tableau, and Power BI; ML with scikit-learn."
    if not jd_text: jd_text = "Looking for a Data Analyst skilled in Python, SQL, Tableau/Power BI; ETL experience preferred."

go = st.button("🚀 Check Match", use_container_width=True)

# ---------------- Inference ----------------
if go:
    if not resume_text or not jd_text:
        st.error("Please provide both Resume and Job Description (paste or upload for each).")
        st.stop()

    model = load_model()
    r_txt, j_txt = clean(resume_text), clean(jd_text)

    R,J = embed(model,[r_txt,j_txt])
    overall = cos_sim(R,J)

    jd_sents = split_sentences(j_txt)[:160]
    r_sents = split_sentences(r_txt)[:500]
    if not jd_sents or not r_sents:
        st.warning("One of the texts has no extractable sentences.")
        st.stop()

    E_jd, E_r = embed(model,jd_sents), embed(model,r_sents)
    sims = np.matmul(E_jd,E_r.T)

    pairs=[(jd_sents[i],r_sents[int(np.argmax(row))],float(max(row))) for i,row in enumerate(sims)]
    df=pd.DataFrame(sorted(pairs,key=lambda x:-x[2])[:top_k],columns=["JD Sentence","Resume Sentence","Similarity"])

    kw_list=[k.strip().lower() for k in kw_text.split(",") if k.strip()]
    r_low, j_low = r_txt.lower(), j_txt.lower()
    have=[k for k in kw_list if k in r_low]
    miss=[k for k in kw_list if k in j_low and k not in r_low]

    k1,k2=st.columns([1,3])
    with k1:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.caption("Overall Fit (cosine)")
        st.markdown(f'<div class="kpi-big">{overall:.3f}</div>',unsafe_allow_html=True)
        pct=max(min((overall-0.5)/0.5,1),0)*100
        st.markdown(f'<div class="progress-outer"><div class="progress-inner" style="width:{pct:.0f}%"></div></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.caption("Quick Tips")
        st.write("• Aim for **0.75+** for strong matches")
        st.write("• Add **missing** but truthful keywords")
        st.write("• Adjust **Profile** and **Domain** if needed")
        st.markdown('</div>',unsafe_allow_html=True)

    st.write("")
    tab1,tab2,tab3=st.tabs(["✅ Skills","🔍 Evidence","🧾 Raw Text"])
    with tab1:
        c1,c2=st.columns(2)
        with c1:
            st.subheader("Found in Resume")
            if have: st.markdown('<div class="chips">'+ " ".join([f"<span>{h}</span>" for h in sorted(set(have))])+"</div>",unsafe_allow_html=True)
            else: st.write("—")
        with c2:
            st.subheader("Missing but in JD")
            if miss: st.markdown('<div class="chips">'+ " ".join([f"<span>{m}</span>" for m in sorted(set(miss))])+"</div>",unsafe_allow_html=True)
            else: st.write("—")
    with tab2:
        st.subheader("Top JD ↔ Resume Matches")
        def _shade(v):
            a=max(min((v-0.6)/0.35,1),0)
            return f"background-color: rgba(124,58,237,{a:.18});"
        styled=df.style.format({"Similarity":"{:.3f}"}).applymap(_shade,subset=["Similarity"])
        st.dataframe(styled,use_container_width=True)
    with tab3:
        st.subheader("Extracted Texts")
        st.text_area("Resume",r_txt,height=200)
        st.text_area("Job Description",j_txt,height=200)

st.markdown('<div class="footer">Made with ✍️ Streamlit + Sentence-BERT · Title uses IBM Plex Mono (typewriter style). Click the title to open your link.</div>',unsafe_allow_html=True)
