"""Promo Finder web app.  Run locally:  streamlit run app.py

Upload a show's screenplay files, pick models, get a ranked promo list. Same pipeline as
find_promos.py. API keys typed here are used for this run only and never stored.
"""
from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from find_promos import promo_dict, seg_str
from promo_finder.candidates import Show
from promo_finder.models import make_llm
from promo_finder.parse import load_pack, validate
from promo_finder.pipeline import run

SAMPLE = Path(__file__).parent / "examples" / "sample_show"
PICKERS = {
    "Groq GPT-OSS 120B → Gemini 3.6 Flash → Gemini 3.5 Flash":
        "groq:openai/gpt-oss-120b,gemini:gemini-3.6-flash,gemini:gemini-3.5-flash",
    "Gemini 3.6 Flash → Gemini 3.5 Flash": "gemini:gemini-3.6-flash,gemini:gemini-3.5-flash",
    "Groq GPT-OSS 120B": "groq:openai/gpt-oss-120b",
    "Offline demo model (no key, not real picks)": "mock",
}
JUDGES = {
    "Gemini 3 Flash → Gemini 3.5 Flash → Groq GPT-OSS (fallback)":
        "gemini:gemini-3-flash-preview,gemini:gemini-3.5-flash,groq:openai/gpt-oss-120b",
    "Gemini 3 Flash (preview)": "gemini:gemini-3-flash-preview",
    "Gemini 3.5 Flash": "gemini:gemini-3.5-flash",
    "Groq GPT-OSS 120B": "groq:openai/gpt-oss-120b",
    "Offline demo model (no key)": "mock",
}

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")
st.title("🎬 Promo Finder")
st.caption("Finds 20–60s promo cuts in microdrama screenplays: exact timestamps, hook, cliffhanger, "
           "why it works, ranked. By Sahaj Yadav · "
           "[code](https://github.com/Raosahaj/Kuku-Task)")

# ------------------------------------------------------------------ inputs ---
with st.sidebar:
    st.header("1 · Screenplays")
    source = st.radio("Input", ["Made-up sample show (2 episodes)", "Upload my own"], index=0)
    files = None
    if source == "Upload my own":
        files = st.file_uploader("Episode .txt files, or one .zip of them", type=["txt", "zip"],
                                 accept_multiple_files=True)
    st.header("2 · Models")
    picker = st.selectbox("Picker (falls back left → right when a quota runs out)", list(PICKERS))
    judge = st.selectbox("Judge (should differ from the picker)", list(JUDGES))
    st.header("3 · API keys")
    st.caption("Used for this run only, never stored. Free keys: "
               "[Gemini](https://aistudio.google.com/apikey) · [Groq](https://console.groq.com/keys)")
    keys = {"GEMINI_API_KEY": st.text_input("Gemini API key", type="password"),
            "GROQ_API_KEY": st.text_input("Groq API key", type="password")}
    keys = {k: v.strip() for k, v in keys.items() if v.strip()}
    episodes = st.text_input("Episodes (blank = all, e.g. 1-5)", "")
    go = st.button("Find promos", type="primary", width="stretch")


def load_inputs(tmp: Path) -> Path:
    """Write uploads (txt or zip) into a temp folder and return it."""
    if source != "Upload my own":
        return SAMPLE
    if not files:
        st.warning("Upload at least one episode file, or pick the sample show.")
        st.stop()
    for f in files:
        data = f.getvalue()
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for name in z.namelist():
                    if name.lower().endswith(".txt") and not name.startswith("__MACOSX"):
                        (tmp / Path(name).name).write_bytes(z.read(name))
        else:
            (tmp / Path(f.name).name).write_bytes(data)
    return tmp


def parse_range(s: str) -> set[int] | None:
    if not s.strip():
        return None
    out = set()
    for part in s.split(","):
        a, _, b = part.strip().partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


# --------------------------------------------------------------------- run ---
if go:
    pick_spec, judge_spec = PICKERS[picker], JUDGES[judge]
    needs = {s.split(":")[0] for s in (pick_spec + "," + judge_spec).split(",")}
    if not keys and needs - {"mock"}:
        st.error("Add a Gemini and/or Groq key in the sidebar, or choose the offline demo models.")
        st.stop()
    with tempfile.TemporaryDirectory() as td:
        folder = load_inputs(Path(td))
        eps = load_pack(folder)
        if not eps or not any(e.events for e in eps):
            st.error("No screenplay lines found. Check the files use the timestamped format "
                     "(see examples/sample_show in the repo).")
            st.stop()
        problems = validate(eps)
        keep = parse_range(episodes)
        sel = [e for e in eps if keep is None or e.episode in keep]
        show = Show(eps)
        st.info(f"Parsed {len(sel)} episodes, {sum(len(e.events) for e in sel)} lines, "
                f"{sum(e.duration for e in sel) / 60:.1f} min of runtime"
                + (f" · {len(problems)} format warnings" if problems else ""))
        llm = make_llm(pick_spec, keys=keys)
        jl = make_llm(judge_spec, keys=keys)
        with st.status("Finding promos…", expanded=True) as status:
            res = run(show, sel, folder.name, llm, jl, workers=1,
                      log=lambda m: status.write(m.strip()), all_eps=eps)
            status.update(label="Done", state="complete")
        st.session_state["result"] = {
            "show": show, "res": res,
            "rows": [promo_dict(show, p) for p in res["ranked"]],
            "usage": llm.usages() + jl.usages(),
        }

# ----------------------------------------------------------------- results ---
r = st.session_state.get("result")
if not r:
    st.markdown("Pick the **made-up sample show** and the **offline demo models**, then press "
                "**Find promos** to see the app work without any key. For real picks, add a free "
                "Gemini or Groq key.")
    st.stop()

res, rows = r["res"], r["rows"]
failed = [l["episode"] for l in res.get("logs", []) if "error" in l]
unjudged = sum(bool(p.scores.get("judge_error")) for p in res["ranked"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Promos", len(rows))
c2.metric("Rejected by rule checks", len(res["rejected"]))
c3.metric("Episodes with no pick", len(failed))
c4.metric("Not judged", unjudged)
if failed:
    st.warning(f"Model calls failed for episodes {failed} (usually a free-tier quota). "
               "Press Find promos again later; finished episodes are cached.")

table = pd.DataFrame([{
    "Rank": x["rank"], "Score": x["score"], "Cut": seg_str(p), "Sec": x["duration_s"],
    "Hook (first 3s)": x["hook"], "Cliffhanger (the cut)": x["cliffhanger"],
    "Why it works": x["why"], "Why this rank": x["rank_reason"],
} for x, p in zip(rows, res["ranked"])])
st.subheader("Ranked promos")
st.dataframe(table, hide_index=True, width="stretch")
st.download_button("Download promos.csv", table.to_csv(index=False).encode("utf-8"),
                   "promos.csv", "text/csv")

st.subheader("Each cut, as a viewer would see it")
for x in rows[:15]:
    cut = " + ".join(f"E{s['episode']:02d} {s['start']}–{s['end']}" for s in x["segments"])
    with st.expander(f"#{x['rank']} · {x['title'] or 'untitled'} · {cut}"):
        st.markdown(f"**Hook:** {x['hook']}  \n**Cliffhanger:** {x['cliffhanger']}  \n"
                    f"**Why it works:** {x['why']}  \n**Why this rank:** {x['rank_reason']}")
        st.code(x["transcript"], language=None)

if res["rejected"]:
    with st.expander(f"Rejected by the rule checker ({len(res['rejected'])})"):
        st.dataframe(pd.DataFrame([{"Cut": seg_str(p) if p.segments else "", "Title": p.title,
                                    "Why rejected": "; ".join(p.issues)} for p in res["rejected"]]),
                     hide_index=True, width="stretch")
with st.expander("Model usage"):
    st.dataframe(pd.DataFrame(r["usage"]), hide_index=True)
