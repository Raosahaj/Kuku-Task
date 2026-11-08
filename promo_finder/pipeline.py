"""Steps 3-4: propose -> validate/repair -> montage -> judge -> rank.

What the LLM does: read scripts, pick moments, judge them as a stranger would.
What code does: every timestamp, every rule check, dedupe, ranking, all outputs.
The model never writes a timestamp; it only names beat IDs.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

from . import prompts as P
from .candidates import HOOK_WINDOW, Promo, Segment, Show, beat_id, parse_beat_id, render_episode
from .models import LLM
from .parse import Episode, fmt

def _obj(resp, key: str) -> dict:
    """Models sometimes drop the JSON wrapper and return the bare list. Normalise."""
    if isinstance(resp, list):
        return {key: resp}
    return resp if isinstance(resp, dict) else {}


# ------------------------------------------------------------------- bible ---


def build_bible(llm: LLM, eps: list[Episode], title: str) -> dict:
    lines = [f"Ep {e.episode}: {e.logline} | NOTES: {e.continuity}" for e in eps]
    try:
        b = _obj(llm.json(P.BIBLE_SYSTEM, P.BIBLE_USER.format(title=title, episodes="\n".join(lines))),
                 "characters")
    except Exception as e:  # bible is helpful, not essential
        b = {"premise": "", "characters": [], "reveals": [], "error": str(e)}
    b.setdefault("premise", "")
    b.setdefault("characters", [])
    b.setdefault("reveals", [])
    return b


def _chars(bible: dict) -> str:
    out = []
    for c in bible.get("characters", [])[:12]:
        al = f" (also {', '.join(c.get('aliases', []))})" if c.get("aliases") else ""
        out.append(f"{c.get('name')}{al}: {c.get('who', '')}")
    return "; ".join(out) or "n/a"


# ------------------------------------------------------- proposal handling ---


def _sim(a: str, b: str) -> float:
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower())
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def to_promo(show: Show, d: dict, source: str) -> Promo:
    """Convert one model pick (beat IDs) into a measured Promo. Never trusts the
    model's own quotes or timings: both are re-derived from the script."""
    p = Promo(segments=[], title=d.get("title", ""), why=d.get("why", ""), source=source)
    p.scores["context_needed"] = d.get("context_needed", "")
    try:
        for s in d.get("segments", []):
            p.segments.append(show.segment(s["first_beat"], s["last_beat"]))
    except (KeyError, ValueError, IndexError) as e:
        p.issues.append(f"invalid beat reference: {e}")
        return p
    p.issues += show.check(p)

    # hook: must exist, must be in the first segment, must start within 3s
    try:
        h = show.event(d["hook_beat"])
        p.hook = h.text
        s0 = p.segments[0]
        if h.episode != s0.episode or not (s0.start - 0.05 <= h.start <= s0.start + HOOK_WINDOW):
            p.issues.append(f"hook beat {d['hook_beat']} not in first {HOOK_WINDOW:.0f}s")
        if d.get("hook") and _sim(d["hook"], h.text) < 0.6:
            p.scores["quote_mismatch"] = True   # model misquoted; we keep the real text
    except (KeyError, ValueError, IndexError):
        p.issues.append("missing/invalid hook_beat")
    # cliffhanger: must be the end of the last segment (no payoff after it)
    try:
        c = show.event(d["cliffhanger_beat"])
        p.cliffhanger = c.text
        sl = p.segments[-1]
        if c.episode != sl.episode or not (sl.end - 4.0 <= c.end <= sl.end + 0.05):
            p.issues.append(f"cliffhanger beat {d['cliffhanger_beat']} is not at the cut")
    except (KeyError, ValueError, IndexError):
        p.issues.append("missing/invalid cliffhanger_beat")
    return p


def propose_episode(llm: LLM, show: Show, ep: Episode, bible: dict, title: str, n_eps: int):
    user = P.PROPOSE_USER.format(title=title, premise=bible.get("premise", ""),
                                 characters=_chars(bible), episode=ep.episode,
                                 n_episodes=n_eps, script=render_episode(ep))
    log = {"episode": ep.episode, "rounds": []}
    try:
        resp = _obj(llm.json(P.PROPOSE_SYSTEM, user), "promos")
    except Exception as e:
        log["error"] = str(e)
        return [], [], [], log
    log["model"] = getattr(llm, "last_spec", llm.spec)   # which model answered (fallback chains)
    picks = [to_promo(show, d, f"ep{ep.episode}") for d in resp.get("promos", [])]
    for p in picks:
        p.scores["proposer"] = log["model"]
    log["rounds"].append({"raw": resp, "issues": [p.issues for p in picks]})

    bad = [(i, p) for i, p in enumerate(picks) if p.issues]
    if bad:  # one repair round with measured problems fed back
        problems = "\n".join(f"- promo {i + 1} ({p.title}): " + "; ".join(p.issues) for i, p in bad)
        try:
            resp2 = _obj(llm.json(P.PROPOSE_SYSTEM, P.REPAIR_USER.format(
                original=user + "\n\nYOUR PREVIOUS ANSWER:\n" + json.dumps(resp), problems=problems)),
                "promos")
            picks2 = [to_promo(show, d, f"ep{ep.episode}:repaired") for d in resp2.get("promos", [])]
            for p in picks2:
                p.scores["proposer"] = getattr(llm, "last_spec", llm.spec)
            log["rounds"].append({"raw": resp2, "issues": [p.issues for p in picks2]})
            good_before = [p for p in picks if not p.issues]
            picks = good_before + [p for p in picks2 if not p.issues and not _dup(p, good_before)] \
                + [p for p in picks2 if p.issues]
            resp = {**resp2, "moments": resp.get("moments", []) + resp2.get("moments", [])}
        except Exception as e:
            log["repair_error"] = str(e)
    ok = [p for p in picks if not p.issues]
    rejected = [p for p in picks if p.issues]
    moments = []
    for m in resp.get("moments", []):
        try:
            show.event(m["beat"])
            moments.append(m)
        except Exception:
            pass
    return ok, rejected, moments, log


def _overlap(a: Promo, b: Promo) -> float:
    tot = 0.0
    for x in a.segments:
        for y in b.segments:
            if x.episode == y.episode:
                tot += max(0.0, min(x.end, y.end) - max(x.start, y.start))
    return tot / max(1e-6, min(a.duration, b.duration))


def _dup(p: Promo, others: list[Promo], thr: float = 0.5) -> bool:
    return any(_overlap(p, o) > thr for o in others)


# ----------------------------------------------------------------- montage ---


def montage(llm: LLM, show: Show, moments: list[dict], bible: dict, title: str, k: int = 5):
    if k <= 0 or len(moments) < 2:
        return [], []
    blocks = []
    for m in moments[:40]:
        ep, idx = parse_beat_id(m["beat"])
        evs = show.eps[ep].events[max(0, idx - 6): idx + 7]
        ctx = "\n".join(f"  {beat_id(ep, e.idx)} [{fmt(e.start)}-{fmt(e.end)}] "
                        f"{e.speaker or e.kind}: {e.text}" for e in evs)
        blocks.append(f"* Ep {ep} {m.get('type', '')} @ {m['beat']}: {m.get('text', '')}\n{ctx}")
    try:
        resp = _obj(llm.json(P.PROPOSE_SYSTEM, P.MONTAGE_USER.format(
            title=title, premise=bible.get("premise", ""), moments="\n".join(blocks), k=k)), "promos")
    except Exception:
        return [], []
    picks = [to_promo(show, d, "montage") for d in resp.get("promos", [])]
    for p in picks:
        if len(p.episodes) < 2 and not p.issues:
            p.issues.append("montage uses a single episode")
    return [p for p in picks if not p.issues], [p for p in picks if p.issues]


# ------------------------------------------------------------------- judge ---


def transcript(show: Show, p: Promo) -> str:
    """What a viewer actually sees/hears, with promo-relative times."""
    out, offset = [], 0.0
    t0 = p.segments[0].start
    for i, s in enumerate(p.segments):
        if len(p.segments) > 1:
            out.append(f"  -- cut {i + 1} --")
        for e in show.eps[s.episode].events:
            if e.start < s.start - 0.05 or e.end > s.end + 0.05:
                continue
            rel = offset + e.start - s.start
            mark = ">>" if (i == 0 and e.start - t0 <= HOOK_WINDOW) else "  "
            who = {"camera": "[visual]", "action": "[action]", "sound": "[sound]"}.get(
                e.kind, e.speaker + ":")
            out.append(f"{mark} {rel:5.1f}s {who} {e.text}")
        offset += s.dur
    return "\n".join(out)


def _judge_chunk(llm: LLM, show: Show, chunk: list[Promo], reveals: str):
    body = "\n\n".join(f"PROMO {j} ({c.duration:.0f}s)\n{transcript(show, c)}"
                       for j, c in enumerate(chunk))
    err = None
    for _attempt in range(2):          # one retry: preview models sometimes return nothing
        try:
            resp = llm.json(P.JUDGE_SYSTEM, P.JUDGE_USER.format(reveals=reveals, promos=body))
            rows = resp.get("scores", []) if isinstance(resp, dict) else resp  # some models drop the wrapper
            return {int(s["id"]): s for s in rows if isinstance(s, dict) and "id" in s}, None
        except Exception as e:
            err = e
            if "blockReason" in str(e):   # a safety block won't change on retry
                break
    return {}, err


def judge(llm: LLM, show: Show, promos: list[Promo], bible: dict, batch: int = 8) -> None:
    """Score promos in batches. If a whole batch fails (e.g. one dark scene trips the
    judge model's safety filter), judge its promos one by one so only the offender is lost."""
    reveals = "\n".join(f"  - ep {r.get('episode')}: {r.get('reveal')}"
                        for r in bible.get("reveals", [])[:15]) or "  (none listed)"
    for i in range(0, len(promos), batch):
        chunk = promos[i:i + batch]
        by_id, err = _judge_chunk(llm, show, chunk, reveals)
        if not by_id and len(chunk) > 1:
            results = [_judge_chunk(llm, show, [c], reveals) for c in chunk]
        else:
            results = [({0: by_id[j]} if j in by_id else {}, err) for j in range(len(chunk))]
        for c, (got, e) in zip(chunk, results):
            s = got.get(0)
            if not s:
                msg = str(e) if e else "judge returned no score for this promo"
                if "blockReason" in msg:
                    msg = "blocked by the judge model's safety filter: check ad-platform policy by hand"
                c.scores["judge_error"] = msg[:200]
                continue
            c.scores.pop("judge_error", None)
            for k in ("hook", "cliffhanger", "standalone", "spoiler"):
                try:
                    c.scores[k] = max(0, min(5, float(s.get(k, 0))))
                except (TypeError, ValueError):
                    pass
            c.scores["judge_note"] = s.get("note", "")


# -------------------------------------------------------------------- rank ---

WEIGHTS = {"hook": 0.35, "cliffhanger": 0.30, "standalone": 0.25, "early": 0.10}


def rank(promos: list[Promo], n_eps: int) -> list[Promo]:
    """Score = weighted judge scores + a nudge for early episodes (a new viewer
    who downloads lands on ep 1), minus spoilers. Posting order then spreads
    picks across episodes so the first week isn't five cuts of the same scene."""
    for p in promos:
        s = p.scores
        early = 5 - 4 * (min(p.episodes) - 1) / max(1, n_eps - 1)
        raw = sum(WEIGHTS[k] * s.get(k, 0) for k in ("hook", "cliffhanger", "standalone"))
        raw += WEIGHTS["early"] * early - 0.4 * s.get("spoiler", 0)
        s["score"] = round(100 * max(0.0, raw) / 5, 1)
    pool = sorted(promos, key=lambda p: -p.scores["score"])
    order, used = [], {}
    while pool:
        best = max(pool, key=lambda p: p.scores["score"] - 6 * sum(used.get(e, 0) for e in p.episodes))
        repeats = sorted(e for e in best.episodes if used.get(e))
        best.scores["repeats_episodes"] = repeats   # explains why rank != pure score order
        pool.remove(best)
        for e in best.episodes:
            used[e] = used.get(e, 0) + 1
        order.append(best)
    for i, p in enumerate(order, 1):
        p.scores["rank"] = i
    return order


def rank_reason(p: Promo) -> str:
    s = p.scores
    if s.get("judge_error"):
        return f"NOT JUDGED (judge failed: {s['judge_error'][:80]}); rank is meaningless, review by hand"
    parts = []
    if s.get("hook", 0) >= 4:
        parts.append("scroll-stopping open")
    if s.get("cliffhanger", 0) >= 4:
        parts.append("strong unresolved cut")
    if s.get("standalone", 0) >= 4:
        parts.append("clear to a cold viewer")
    if min(p.episodes) <= 5:
        parts.append(f"lands viewers near ep {min(p.episodes)}")
    if s.get("spoiler", 0) >= 1:
        parts.append("some spoiler risk")
    if s.get("repeats_episodes"):
        eps = ", ".join(str(e) for e in s["repeats_episodes"])
        parts.append(f"posted later because ep {eps} already has a higher-ranked promo")
    weak = s.get("judge_note")
    return "; ".join(parts) + (f". Weakness: {weak}" if weak else "")


# ---------------------------------------------------------------- run all ---


def run(show: Show, eps: list[Episode], title: str, llm: LLM, judge_llm: LLM,
        workers: int = 2, montage_k: int = 5, log=print, all_eps: list[Episode] | None = None):
    n_eps = max(show.eps) if show.eps else 0
    log(f"[1/4] show bible ({llm.spec})")
    bible = build_bible(llm, all_eps or eps, title)   # always the whole show, even for a subset run
    log(f"[2/4] proposing promos for {len(eps)} episodes")
    ok, rejected, moments, logs = [], [], [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for good, bad, mom, lg in ex.map(lambda e: propose_episode(llm, show, e, bible, title, n_eps), eps):
            ok += good
            rejected += bad
            moments += [{**m, "episode": lg["episode"]} for m in mom]
            logs.append(lg)
            log(f"   ep{lg['episode']:>3}: {len(good)} ok, {len(bad)} rejected"
                + (f"  ERROR {lg['error'][:80]}" if "error" in lg else ""))
    log(f"[3/4] cross-episode montage from {len(moments)} moments")
    mok, mbad = montage(llm, show, moments, bible, title, montage_k)
    rejected += mbad
    kept = []
    for p in ok + mok:      # drop near-duplicates (same footage)
        if not _dup(p, kept):
            kept.append(p)
    log(f"[4/4] judging {len(kept)} promos with {judge_llm.spec}")
    judge(judge_llm, show, kept, bible)
    ranked = rank(kept, n_eps)
    return {"bible": bible, "ranked": ranked, "rejected": rejected, "moments": moments, "logs": logs}
