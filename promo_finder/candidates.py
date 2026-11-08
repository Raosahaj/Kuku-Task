"""Step 2: turn line-level picks into clean, rule-checked promo segments.

Everything here is deterministic code:
  * numbered "beats" (dialogue/action/sound lines) the LLM can reference by ID
  * snapping segment edges to camera-shot boundaries without cutting a line
  * hard rules every promo must pass (20-60s, order, hook in first 3s, ...)
  * a no-LLM heuristic baseline, used as a fallback and as a yardstick
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parse import Episode, Event, fmt

MIN_LEN, MAX_LEN = 20.0, 60.0
HOOK_WINDOW = 3.0          # hook must start within the first 3s of the promo
MIN_SEGMENT = 4.0          # shorter fragments are unreadable on a feed
SNAP_SLACK = 2.5           # how far we let a cut drift to land on a shot boundary


def beat_id(ep: int, idx: int) -> str:
    return f"E{ep:02d}-{idx:03d}"


def parse_beat_id(bid: str) -> tuple[int, int]:
    m = re.fullmatch(r"E(\d+)-(\d+)", bid.strip())
    if not m:
        raise ValueError(f"bad beat id {bid!r}")
    return int(m.group(1)), int(m.group(2))


def render_episode(ep: Episode, with_camera: bool = True) -> str:
    """Compact, ID-tagged script for the LLM. Camera lines are kept because
    visual hooks (a slap, a reveal) are often only described there."""
    out = [f"EPISODE {ep.episode} ({fmt(ep.duration)}) — {ep.logline}"]
    scene = None
    for ev in ep.events:
        if ev.scene != scene:
            scene = ev.scene
            sc = next((s for s in ep.scenes if s.scene == scene), None)
            out.append(f"\n[SCENE {scene}: {sc.heading if sc else ''}]")
        if ev.kind == "camera" and not with_camera:
            continue
        who = {"dialogue": ev.speaker + (f" ({ev.delivery})" if ev.delivery else ""),
               "action": "ACTION", "camera": "cam", "sound": "SFX"}[ev.kind]
        txt = f'"{ev.text}"' if ev.kind == "dialogue" else ev.text
        out.append(f"{beat_id(ep.episode, ev.idx)} [{fmt(ev.start)}-{fmt(ev.end)}] {who}: {txt}")
    return "\n".join(out)


@dataclass
class Segment:
    episode: int
    start: float
    end: float
    first_beat: str = ""
    last_beat: str = ""

    @property
    def dur(self) -> float:
        return self.end - self.start

    def as_dict(self) -> dict:
        return {"episode": self.episode, "start": fmt(self.start), "end": fmt(self.end),
                "start_s": round(self.start, 2), "end_s": round(self.end, 2),
                "first_beat": self.first_beat, "last_beat": self.last_beat}


@dataclass
class Promo:
    segments: list[Segment]
    hook: str = ""
    cliffhanger: str = ""
    why: str = ""
    title: str = ""
    source: str = ""                       # which proposer produced it
    scores: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(s.dur for s in self.segments)

    @property
    def episodes(self) -> list[int]:
        return sorted({s.episode for s in self.segments})


class Show:
    """Index over parsed episodes for fast lookups and snapping."""

    def __init__(self, eps: list[Episode]):
        self.eps = {e.episode: e for e in eps}

    def event(self, bid: str) -> Event:
        ep, idx = parse_beat_id(bid)
        return self.eps[ep].events[idx]

    def _lines(self, ep: int) -> list[Event]:
        return [e for e in self.eps[ep].events if e.kind == "dialogue"]

    def _shot_edges(self, ep: int) -> list[float]:
        e = self.eps[ep]
        edges = {0.0, e.duration}
        for ev in e.events:
            if ev.kind == "camera":
                edges.update((ev.start, ev.end))
        for s in e.scenes:
            edges.update((s.start, s.end))
        return sorted(edges)

    def _cuts_line(self, ep: int, t: float) -> bool:
        return any(l.start + 0.05 < t < l.end - 0.05 for l in self._lines(ep))

    def snap_start(self, ep: int, t: float) -> float:
        """Latest shot boundary at or before t (within slack) that cuts no line."""
        for edge in reversed(self._shot_edges(ep)):
            if edge <= t + 1e-6 and t - edge <= SNAP_SLACK and not self._cuts_line(ep, edge):
                return edge
        return self._clear_line(ep, t, backward=True)

    def _clear_line(self, ep: int, t: float, backward: bool) -> float:
        """Fallback: move outward until t no longer sits inside a spoken line."""
        for _ in range(10):
            hit = [l for l in self._lines(ep) if l.start + 0.05 < t < l.end - 0.05]
            if not hit:
                return t
            t = min(l.start for l in hit) if backward else max(l.end for l in hit)
        return t

    def snap_end(self, ep: int, t: float) -> float:
        """Earliest shot boundary at or after t (within slack) that cuts no line."""
        for edge in self._shot_edges(ep):
            if edge >= t - 1e-6 and edge - t <= SNAP_SLACK and not self._cuts_line(ep, edge):
                return edge
        return self._clear_line(ep, t, backward=False)

    def segment(self, first: str, last: str, snap: bool = True) -> Segment:
        a, b = self.event(first), self.event(last)
        if a.episode != b.episode:
            raise ValueError(f"segment spans episodes: {first}..{last}")
        if b.end < a.start:
            a, b, first, last = b, a, last, first
        start, end = a.start, b.end
        if snap:
            start, end = self.snap_start(a.episode, start), self.snap_end(a.episode, end)
        return Segment(a.episode, start, end, first, last)

    def text_between(self, seg: Segment, kinds=("dialogue", "action", "sound")) -> list[Event]:
        return [e for e in self.eps[seg.episode].events
                if e.kind in kinds and e.start >= seg.start - 0.05 and e.end <= seg.end + 0.05]

    def check(self, p: Promo) -> list[str]:
        """Hard rules. A promo with issues is dropped or repaired upstream."""
        issues = []
        if not p.segments:
            return ["no segments"]
        if not (MIN_LEN - 0.5 <= p.duration <= MAX_LEN + 0.5):
            issues.append(f"duration {p.duration:.1f}s outside {MIN_LEN:.0f}-{MAX_LEN:.0f}s")
        for s in p.segments:
            ep = self.eps.get(s.episode)
            if ep is None:
                issues.append(f"unknown episode {s.episode}")
                continue
            if s.start < -0.01 or s.end > ep.duration + 0.01 or s.end <= s.start:
                issues.append(f"ep{s.episode} {fmt(s.start)}-{fmt(s.end)} out of bounds")
            if len(p.segments) > 1 and s.dur < MIN_SEGMENT:
                issues.append(f"segment ep{s.episode} {fmt(s.start)} only {s.dur:.1f}s")
            for t in (s.start, s.end):
                if self._cuts_line(s.episode, t):
                    issues.append(f"ep{s.episode} cut at {fmt(t)} lands mid-line")
        for x, y in zip(p.segments, p.segments[1:]):
            if x.episode == y.episode and x.end > y.start and y.end > x.start:
                issues.append(f"overlapping segments in ep{x.episode}")
        # hook placement: first spoken/acted beat must begin within 3s
        first = self.text_between(p.segments[0])
        if first and first[0].start - p.segments[0].start > HOOK_WINDOW:
            issues.append(f"first beat starts {first[0].start - p.segments[0].start:.1f}s in (dead air)")
        return issues


# ---------------------------------------------------------------- baseline ---
HOOK_WORDS = re.compile(
    r"\b(slap|strik|punch|grab|shove|slam|kneel|collapse|faint|blood|knife|sword|gun|"
    r"kiss|tear|rip|scream|gasp|reveal|secret|pregnan|dead|die|kill|betray|arrest|"
    r"divorce|marry|married|wedding|poison|throw|burn|crash|fall)\w*", re.I)
CLIFF_WORDS = re.compile(r"\b(cliffhanger|cut to black|freeze|ominous|stinger|shock)\w*", re.I)


def hook_score(ev: Event) -> float:
    s = 0.0
    if ev.kind == "dialogue":
        s += 1.0 * ev.text.count("!") + 0.6 * ("?" in ev.text)
        s += 0.8 * bool(re.search(r"(shout|scream|furious|shock|outrag|cold|cruel|threat)",
                                  ev.delivery, re.I))
    if ev.kind in ("action", "camera"):
        s += 0.5
    s += 1.2 * len(HOOK_WORDS.findall(ev.text))
    return s


def cliff_score(ev: Event, ep: Episode) -> float:
    s = 2.0 * len(CLIFF_WORDS.findall(ev.text))
    if ev.kind == "dialogue":
        s += 1.0 * ev.text.rstrip().endswith(("?", "...", "—", "-")) + 0.6 * ("!" in ev.text)
    if ep.duration - ev.end < 6:
        s += 1.5            # writers already put a cliffhanger at the episode end
    return s


def baseline_promos(show: Show, per_episode: int = 2) -> list[Promo]:
    """Keyword/structure heuristic. No model. Deliberately simple."""
    out = []
    for ep in show.eps.values():
        beats = [e for e in ep.events if e.kind != "camera" or HOOK_WORDS.search(e.text)]
        cands = []
        for a in beats:
            hs = hook_score(a)
            if hs < 1.5:
                continue
            for b in beats:
                d = b.end - a.start
                if MIN_LEN <= d <= MAX_LEN and b.kind != "camera":
                    cands.append((hs + cliff_score(b, ep) - 0.01 * d, a, b))
        cands.sort(key=lambda c: -c[0])
        taken: list[Segment] = []
        for sc, a, b in cands:
            seg = show.segment(beat_id(ep.episode, a.idx), beat_id(ep.episode, b.idx))
            if any(seg.start < t.end and t.start < seg.end for t in taken):
                continue
            p = Promo([seg], hook=a.text, cliffhanger=b.text,
                      why="heuristic: strong keywords at open, unresolved beat at close",
                      source="baseline", scores={"baseline": round(sc, 2)})
            p.issues = show.check(p)
            if not p.issues:
                taken.append(seg)
                out.append(p)
            if len(taken) >= per_episode:
                break
    return out
