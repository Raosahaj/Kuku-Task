#!/usr/bin/env python3
"""Quality checks on one or two runs.

  python evaluate.py --run outputs/run                       # rule stats + gold recall
  python evaluate.py --run outputs/run --compare outputs/run2  # stability between runs
  python evaluate.py --run outputs/run --compare outputs/baseline

gold.csv (hand-labelled by an editor): episode,start,end,note   times as MM:SS.xx
A gold promo counts as FOUND if any tool promo overlaps it by >=50% of the shorter clip.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path


def sec(t: str) -> float:
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def load(run: Path) -> list[dict]:
    return json.loads((run / "promos.json").read_text(encoding="utf-8"))


def spans(p: dict) -> list[tuple[int, float, float]]:
    return [(s["episode"], s["start_s"], s["end_s"]) for s in p["segments"]]


def overlap(a, b) -> float:
    tot = sum(max(0, min(e1, e2) - max(s1, s2)) for x1, s1, e1 in a for x2, s2, e2 in b if x1 == x2)
    la, lb = sum(e - s for _, s, e in a), sum(e - s for _, s, e in b)
    return tot / max(1e-6, min(la, lb))


def matched(A, B, thr=0.5) -> int:
    return sum(any(overlap(spans(a), spans(b)) >= thr for b in B) for a in A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--compare")
    ap.add_argument("--gold", default="gold.csv")
    ap.add_argument("--top", type=int, default=20, help="'top-N' cutoff for comparisons")
    a = ap.parse_args()
    run = Path(a.run)
    P = load(run)
    rep = {"run": str(run), "n_promos": len(P)}

    rej = json.loads((run / "rejected.json").read_text(encoding="utf-8")) if (run / "rejected.json").exists() else []
    rep["rejected_by_rules"] = len(rej)
    reasons = {}
    for r in rej:
        for i in r["issues"]:
            k = i.split(":")[0].split(" ")[0] if i.startswith("invalid") else " ".join(i.split()[:3])
            reasons[k] = reasons.get(k, 0) + 1
    rep["top_rejection_reasons"] = dict(sorted(reasons.items(), key=lambda x: -x[1])[:6])
    logs = run / "logs.json"
    if logs.exists():
        L = json.loads(logs.read_text(encoding="utf-8"))
        rep["episodes_needing_repair"] = sum(len(l.get("rounds", [])) > 1 for l in L)
        rep["episodes_with_errors"] = [l["episode"] for l in L if "error" in l]
    if P:
        rep["duration_s"] = {"min": min(p["duration_s"] for p in P),
                             "median": st.median(p["duration_s"] for p in P),
                             "max": max(p["duration_s"] for p in P)}
        rep["multi_segment"] = sum(len(p["segments"]) > 1 for p in P)
        rep["episodes_covered"] = len({e for p in P for e in p["episodes"]})
        rep["quote_mismatches"] = sum(bool(p["scores"].get("quote_mismatch")) for p in P)
        for k in ("hook", "cliffhanger", "standalone"):
            v = [p["scores"][k] for p in P if isinstance(p["scores"].get(k), (int, float))]
            if v:
                rep[f"mean_{k}"] = round(st.mean(v), 2)
        rep["low_standalone(<=2)"] = sum(p["scores"].get("standalone", 5) <= 2 for p in P)

    gold = Path(a.gold)
    if gold.exists():
        G = [{"segments": [{"episode": int(r["episode"]), "start_s": sec(r["start"]),
                            "end_s": sec(r["end"])}], "note": r.get("note", "")}
             for r in csv.DictReader(open(gold, encoding="utf-8-sig"))]
        found = [g for g in G if any(overlap(spans(g), spans(p)) >= 0.5 for p in P)]
        topn = P[: a.top]
        rep["gold"] = {"n": len(G), "recall_all": f"{len(found)}/{len(G)}",
                       f"recall_top{a.top}": f"{sum(any(overlap(spans(g), spans(p)) >= .5 for p in topn) for g in G)}/{len(G)}",
                       "missed": [g["note"] or g["segments"][0] for g in G if g not in found]}

    if a.compare:
        Q = load(Path(a.compare))
        tp, tq = P[: a.top], Q[: a.top]
        rep["compare"] = {"other": a.compare, "n_other": len(Q),
                          f"top{a.top}_overlap": f"{matched(tp, tq)}/{len(tp)}",
                          "all_overlap": f"{matched(P, Q)}/{len(P)}"}
    print(json.dumps(rep, indent=2, default=str))
    (run / "eval.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
