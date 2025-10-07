# MatchMyResume — Stable + Stateful
- Results are persisted in `st.session_state["results"]` so tabs won’t disappear on chat submit
- Hardened tabs (no Pandas Styler), safe serialization, robust chat fallback
- Courier title, modern theme, paste+upload inputs (paste wins)
- Sentence-BERT similarity + evidence pairs
- ATS score (100) with breakdown
- Coach tab: heuristic by default; uses OpenAI if `OPENAI_API_KEY` is set
