import io, re, regex, numpy as np, pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document

st.set_page_config(page_title="MatchMyResume — AI JD Matcher", page_icon="✍️", layout="wide")

# ---------------- CSS (polished) ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@700&family=Inter:wght@400;600;800&display=swap');
:root{--violet:#7C3AED;--violet2:#8B5CF6;--emerald:#22C55E;--panel:#0F172A;--muted:#94A3B8;}
html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
.block-container{max-width:1080px;padding-top:1rem;padding-bottom:2rem;}
/* HERO */
.hero{position:relative;background:var(--panel);border:1px solid rgba(255,255,255,.06);
      border-radius:20px;padding:18px;border-left:3px solid var(--violet);
      box-shadow:0 10px 26px rgba(0,0,0,.25);overflow:hidden;}
.brand{display:flex;align-items:center;gap:14px;position:relative;z-index:2;}
.brand-badge{width:44px;height:44px;border-radius:12px;
  background:linear-gradient(135deg,var(--violet),var(--violet2) 55%,var(--emerald));
  box-shadow:0 10px 26px rgba(124,58,237,.35);}
.brand-title{font-family:'Courier Prime', monospace;font-weight:700;letter-spacing:.4px;
  font-size: clamp(32px, 4vw, 46px);color:#F8FAFC;line-height:1.06;margin:0;white-space:nowrap;}
/* Typewriter effect */
.brand-title .tw{display:inline-block;overflow:hidden;white-space:nowrap;border-right:.12em solid #F8FAFC;
  animation: typing 2.6s steps(20,end), blink .8s step-end infinite;}
@keyframes typing {from{width:0} to{width:100%}}
@keyframes blink {50% {border-color: transparent}}
.brand-sub{margin:.25rem 0 0 0;color:#E2E8F0;}
.pill{display:inline-block;margin-top:6px;padding:.22rem .6rem;border-radius:999px;
  font-size:.78rem;color:#fff;background:rgba(124,58,237,.28);border:1px solid rgba(124,58,237,.38)}
/* Cards & KPI */
.card{background:rgba(148,163,184,.08);border:1px solid rgba(255,255,255,.08);
      border-radius:16px;padding:1rem 1.25rem;}
.kpi-big{font-size:1.3rem;font-weight:800;}
.progress-outer{height:10px;background:rgba(100,116,139,.35);border-radius:8px;overflow:hidden;}
.progress-inner{height:10px;background:linear-gradient(90deg,var(--violet),var(--violet2),var(--emerald));width:0%;}
/* Chips */
.chips span{display:inline-block;margin:.22rem .28rem;padding:.28rem .65rem;border-radius:999px;
  border:1px solid rgba(148,163,184,.4);background:#1F2937;color:#e2e8f0;font-size:.85rem;}
/* Buttons & focus */
.stButton>button{background:linear-gradient(90deg,var(--violet),var(--violet2),var(--emerald));
  color:#fff;border:none;border-radius:12px;padding:.7rem 1.1rem;font-weight:800;outline:2px solid transparent;}
.stButton>button:focus{outline:3px solid #8B5CF6;}
.stButton>button:hover{filter:brightness(1.06);}
/* Footer */
.footer{color:var(--muted);font-size:.92rem;margin-top:2rem;text-align:center;}
.footer a{color:#A5B4FC;text-decoration:none;} .footer a:hover{text-decoration:underline;}
/* Inputs spacing */
section[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,.06); }
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("""
<div class="hero" role="banner" aria-label="App header">
  <div class="brand">
    <div class="brand-badge" aria-hidden="true"></div>
    <div>
      <div class="brand-title"><span class="tw">MatchMyResume</span></div>
      <div class="brand-sub">AI Resume ↔ JD Matcher · Sentence-BERT embeddings</div>
      <div class="pill">Paste or upload · Fit score + evidence · Missing skills</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------- Helpers ----------------
@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def clean(t): return regex.sub(r"\s+", " ", t.replace("\u00A0"," ").strip())

def split_sentences(t):
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', t) if s.strip()]
    # Soft length cap to avoid huge embeddings
    return sents[:600]

def embed(m, texts): return m.encode(texts, normalize_embeddings=True)
def cos_sim(a,b): return float(np.dot(a,b))

def read_any(uploaded_file)->str:
    name = uploaded_file.name.lower()
    if uploaded_file.size and uploaded_file.size > 2_000_000:
        st.warning(f"{uploaded_file.name}: Large file detected (>2MB). Parsing may be slow.")
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        pdf = PdfReader(io.BytesIO(data))
        texts = [p.extract_text() or "" for p in pdf.pages]
        if sum(len(t) for t in texts) == 0:
            st.info("This PDF seems to be scanned (no text). Try DOCX/TXT or OCR.")
        return " ".join(texts)
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join([p.text for p in doc.paragraphs])
    return data.decode("utf-8", errors="ignore")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Controls")

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
    base_kw = ", ".join([x for x in [role_presets[role], domain_presets[domain]] if x])
    kw_text = st.text_area("Keywords (editable, comma-separated)", value=base_kw, height=110)

    auto_kw = st.checkbox("Auto-extract top JD terms (adds 1–2 grams)", value=True)
    top_k = st.slider("Top evidence pairs", 3, 20, 10)
    st.divider()
    if st.button("Reset inputs"):
        for key in ["resume_ta","jd_ta","resume_up","jd_up"]:
            if key in st.session_state: del st.session_state[key]
        st.experimental_rerun()

# ---------------- Inputs (upload + paste) ----------------
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

    # Auto-extract top JD terms (1-2 grams)
    auto_terms = []
    if auto_kw:
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            vec = CountVectorizer(stop_words="english", ngram_range=(1,2), max_features=40)
            _ = vec.fit_transform([j_txt])
            auto_terms = [t.lower() for t in vec.get_feature_names_out() if len(t) > 2]
        except Exception as e:
            st.info("Auto-extract skipped (scikit-learn not available).")

    # Compute overall similarity
    R,J = embed(model,[r_txt,j_txt])
    overall = cos_sim(R,J)

    # Sentence retrieval
    jd_sents = split_sentences(j_txt)
    r_sents = split_sentences(r_txt)
    if not jd_sents or not r_sents:
        st.warning("One of the texts has no extractable sentences.")
        st.stop()

    E_jd, E_r = embed(model,jd_sents), embed(model,r_sents)
    sims = np.matmul(E_jd,E_r.T)

    pairs=[(jd_sents[i],r_sents[int(np.argmax(row))],float(max(row))) for i,row in enumerate(sims)]
    df=pd.DataFrame(sorted(pairs,key=lambda x:-x[2])[:top_k],columns=["JD Sentence","Resume Sentence","Similarity"])

    # Keywords
    manual_kw=[k.strip().lower() for k in kw_text.split(",") if k.strip()]
    kw_list=sorted(set(manual_kw + auto_terms))
    r_low, j_low = r_txt.lower(), j_txt.lower()
    have=[k for k in kw_list if k in r_low]
    miss=[k for k in kw_list if k in j_low and k not in r_low]

    # KPI
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
    tab1,tab2,tab3,tab4=st.tabs(["✅ Skills","🔍 Evidence","🧾 Raw Text","⬇️ Export"])
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

        st.markdown("")
        st.text_area("Copy missing keywords", value=", ".join(sorted(set(miss))), height=80)

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

    with tab4:
        st.subheader("Download results")
        # CSV (evidence + keywords flat)
        out = df.copy()
        out["Have Keywords"] = ", ".join(sorted(set(have)))
        out["Missing Keywords"] = ", ".join(sorted(set(miss)))
        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name="match_results.csv", mime="text/csv")

        # JSON report
        report = {
            "overall_fit": round(overall, 3),
            "profile": role,
            "domain": domain,
            "keywords_used": kw_list,
            "have": sorted(set(have)),
            "missing": sorted(set(miss)),
            "evidence": [{"jd": r["JD Sentence"], "resume": r["Resume Sentence"], "sim": round(float(r["Similarity"]),3)} for _, r in out.iterrows()]
        }
        import json
        json_bytes = json.dumps(report, indent=2).encode("utf-8")
        st.download_button("Download JSON", json_bytes, file_name="match_report.json", mime="application/json")

st.markdown('<div class="footer">Built by <a href="https://github.com/subashchakravarthy" target="_blank">Subash Chakravarthy</a> · © 2025</div>', unsafe_allow_html=True)
