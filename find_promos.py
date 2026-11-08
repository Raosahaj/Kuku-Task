#!/usr/bin/env python3
"""Find the best promos in a microdrama screenplay pack.

  python find_promos.py --input data/ --model gemini:gemini-3.8-flash --judge-model groq:openai/gpt-oss-120b
  python find_promos.py --input data/ --model ollama:qwen2.5:7b --episodes 1-5
  python find_promos.py --input data/ --baseline          # no model, heuristic only
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from promo_finder.candidates import Show, baseline_promos
from promo_finder.models import make_llm
from promo_finder.parse import fmt, load_pack, validate, write_tables
from promo_finder.pipeline import build_bible, judge, rank, rank_reason, run, transcript

# Published list prices (USD per 1M tokens), used only for the cost estimate.
PRICES = {"gemini:gemini-2.5-flash": (0.30, 2.50), "groq:llama-3.3-70b-versatile": (0.59, 0.79),
          "openai:gpt-4o-mini": (0.15, 0.60)}


def parse_range(s: str | None, n: int) -> set[int] | None:
    if not s:
        return None
    out = set()
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def seg_str(p) -> str:
    return " + ".join(f"E{s.episode:02d} {fmt(s.start)}-{fmt(s.end)}" for s in p.segments)


def promo_dict(show, p) -> dict:
    return {"rank": p.scores.get("rank"), "score": p.scores.get("score"), "title": p.title,
            "duration_s": round(p.duration, 1), "episodes": p.episodes,
            "segments": [s.as_dict() for s in p.segments],
            "hook": p.hook, "cliffhanger": p.cliffhanger, "why": p.why,
            "rank_reason": rank_reason(p), "source": p.source,
            "scores": {k: v for k, v in p.scores.items() if k not in ("rank", "score")},
            "issues": p.issues, "transcript": transcript(show, p)}


def write_outputs(out: Path, show, res: dict, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ranked = [promo_dict(show, p) for p in res["ranked"]]
    (out / "promos.json").write_text(json.dumps(ranked, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "rejected.json").write_text(json.dumps(
        [{"segments": [s.as_dict() for s in p.segments], "title": p.title, "source": p.source,
          "issues": p.issues} for p in res["rejected"]], indent=2, ensure_ascii=False), encoding="utf-8")
    for k in ("bible", "moments", "logs"):
        if res.get(k) is not None:
            (out / f"{k}.json").write_text(json.dumps(res[k], indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with open(out / "promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "score", "segments", "duration_s", "hook", "cliffhanger", "why",
                    "rank_reason", "hook_1to5", "cliff_1to5", "standalone_1to5", "spoiler_0to2",
                    "source"])
        for p in res["ranked"]:
            s = p.scores
            w.writerow([s.get("rank"), s.get("score"), seg_str(p), round(p.duration, 1), p.hook,
                        p.cliffhanger, p.why, rank_reason(p), s.get("hook"), s.get("cliffhanger"),
                        s.get("standalone"), s.get("spoiler"), p.source])

    md = [f"# Promos — {meta['title']}", "",
          f"Model `{meta['model']}` · judge `{meta['judge_model']}` · {len(ranked)} promos · "
          f"{len(res['rejected'])} rejected by rule checks", ""]
    for p in res["ranked"]:
        s = p.scores
        md += [f"## #{s['rank']} · {p.title or 'untitled'} · score {s['score']}",
               f"**Cut:** {seg_str(p)} ({p.duration:.0f}s)  ",
               f"**Hook:** {p.hook}  ", f"**Cliffhanger:** {p.cliffhanger}  ",
               f"**Why it works:** {p.why}  ", f"**Why this rank:** {rank_reason(p)}", ""]
    (out / "promos.md").write_text("\n".join(md), encoding="utf-8")


def load_dotenv(path: Path = Path(".env")) -> None:
    """Read KEY=value lines from .env (git-ignored) so API keys never go on the command line."""
    import os
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and k and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder of episode .txt screenplays")
    ap.add_argument("--model", default="gemini:gemini-3.8-flash", help="provider:model, or a comma list to fall back when a free quota runs out")
    ap.add_argument("--judge-model", default=None, help="separate model for judging (default: --model)")
    ap.add_argument("--out", default="outputs/run", help="output folder")
    ap.add_argument("--title", default=None, help="show title (default: input folder name)")
    ap.add_argument("--episodes", default=None, help="subset, e.g. 1-5 or 1,3,10-12")
    ap.add_argument("--workers", type=int, default=2, help="parallel calls (keep low on free tiers)")
    ap.add_argument("--montage", type=int, default=5, help="max cross-episode promos")
    ap.add_argument("--baseline", action="store_true", help="no model: keyword heuristic only")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    inp = Path(a.input)
    eps = load_pack(inp)
    problems = validate(eps)
    if problems:
        print(f"WARNING: {len(problems)} screenplay issues, e.g. {problems[:3]}")
    title = a.title or (next((p.parent.name for p in inp.rglob('*.txt')), inp.name))
    keep = parse_range(a.episodes, len(eps))
    sel = [e for e in eps if keep is None or e.episode in keep]
    show = Show(eps)
    out = Path(a.out)
    write_tables(sel, out / "parsed")
    print(f"Parsed {len(sel)} episodes, {sum(len(e.events) for e in sel)} beats, "
          f"{sum(e.duration for e in sel) / 60:.1f} min of runtime")

    meta = {"title": title, "episodes": [e.episode for e in sel], "parse_issues": problems}
    if a.baseline:
        ranked = baseline_promos(Show(sel))
        if a.judge_model:   # score the heuristic's picks with the same judge, for a fair A/B
            jl = make_llm(a.judge_model, use_cache=not a.no_cache)
            judge(jl, show, ranked, build_bible(jl, eps, title))
            meta.update(usage=jl.usages())
        else:               # crude stand-in scores so the ranking still runs offline
            for p in ranked:
                b = p.scores["baseline"]
                p.scores.update(hook=min(5, b / 2), cliffhanger=min(5, b / 2), standalone=3, spoiler=0)
        res = {"ranked": rank(ranked, max(show.eps)), "rejected": [], "bible": None,
               "moments": None, "logs": None}
        meta.update(model="baseline (no LLM)", judge_model=a.judge_model or "none",
                    est_cost_usd_at_list_price=0.0)
    else:
        llm = make_llm(a.model, use_cache=not a.no_cache)
        jl = make_llm(a.judge_model, use_cache=not a.no_cache) if a.judge_model else llm
        res = run(show, sel, title, llm, jl, workers=a.workers, montage_k=a.montage, all_eps=eps)
        usage = llm.usages() + (jl.usages() if jl is not llm else [])
        cost = 0.0
        for u in usage:
            pin, pout = PRICES.get(u["model"], (0, 0))
            cost += (u["input_tokens"] * pin + u["output_tokens"] * pout) / 1e6
        meta.update(model=a.model, judge_model=a.judge_model or a.model, usage=usage,
                    est_cost_usd_at_list_price=round(cost, 4))
    meta["seconds"] = round(time.time() - t0, 1)
    failed_eps = [l["episode"] for l in (res.get("logs") or []) if "error" in l]
    unjudged = sum(bool(p.scores.get("judge_error")) for p in res["ranked"])
    meta.update(episodes_failed=failed_eps, promos_not_judged=unjudged)
    write_outputs(out, show, res, meta)
    if failed_eps:
        print(f"WARNING: {len(failed_eps)} episodes got no picks because the model call failed: {failed_eps}. "
              "Re-run the same command; finished calls come from the cache.")
    if unjudged:
        print(f"WARNING: {unjudged} promos were not scored by the judge; their rank is meaningless.")
    print(f"\n{len(res['ranked'])} promos ranked, {len(res['rejected'])} rejected -> {out}/")
    for p in res["ranked"][:10]:
        print(f"  #{p.scores['rank']:>2} {p.scores['score']:>5}  {seg_str(p):<34} {p.hook[:55]}")
    if "usage" in meta:
        print("usage:", json.dumps(meta["usage"]), "| est. cost $", meta["est_cost_usd_at_list_price"])


if __name__ == "__main__":
    main()
