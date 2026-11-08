"""Step 1: parse timestamped microdrama screenplays into a flat events table.

Pure code, no AI. One row per timed block (dialogue line, action, camera, sound).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

TS = r"(\d{2}):(\d{2})\.(\d{2})"
RE_BLOCK_TIME = re.compile(rf"^Start: \({TS}\) \| End: \({TS}\)$")
RE_SCENE_TIME = re.compile(rf"^Scene (start|end): {TS}$", re.I)
RE_LABEL = re.compile(r"^\*\*(.+?):\*\*$")
RE_FILE_EP = re.compile(r"episode-(\d+)", re.I)

NON_SPEAKER = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sound"}
VOICE_TAGS = {"O.S.", "V.O.", "O.C.", "CONT'D"}


def to_sec(mm: str, ss: str, cc: str) -> float:
    return int(mm) * 60 + int(ss) + int(cc) / 100


def fmt(t: float) -> str:
    m, s = divmod(round(t, 2), 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Event:
    episode: int
    scene: int
    idx: int            # order within episode
    kind: str           # dialogue | action | camera | sound
    speaker: str        # raw label for dialogue, "" otherwise
    delivery: str       # parenthetical emotion cues, e.g. "Cold, Commanding"
    offscreen: bool
    text: str
    start: float
    end: float


@dataclass
class Scene:
    episode: int
    scene: int
    heading: str
    start: float
    end: float


@dataclass
class Episode:
    episode: int
    file: str
    logline: str = ""
    continuity: str = ""
    duration: float = 0.0
    raw_blocks: int = 0  # timed blocks seen in the file, for a completeness check
    scenes: list[Scene] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)


def _parse_label(label: str) -> tuple[str, str, str, bool]:
    """'MATRIARCH (O.S.) (Firm)' -> (kind, speaker, delivery, offscreen)."""
    if label in NON_SPEAKER:
        return NON_SPEAKER[label], "", "", False
    name = label.split(" (", 1)[0].strip()
    parens = re.findall(r"\(([^)]*)\)", label)
    offscreen = any(p.strip().upper() in VOICE_TAGS for p in parens)
    delivery = ", ".join(p for p in parens if p.strip().upper() not in VOICE_TAGS)
    return "dialogue", name, delivery, offscreen


def parse_file(path: Path) -> Episode:
    lines = path.read_text(encoding="utf-8").splitlines()
    m = RE_FILE_EP.search(path.name)
    ep = Episode(episode=int(m.group(1)) if m else 0, file=path.name)

    scene_no, heading, s_start = 0, "", 0.0
    pending_label: str | None = None
    pending_time: tuple[float, float] | None = None
    idx = 0

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("EPISODE NUMBER:"):
            num = re.search(r"\d+", line)
            if num:
                ep.episode = int(num.group())
        elif line.startswith("EPISODE LOGLINE:"):
            ep.logline = line.split(":", 1)[1].strip()
        elif line.startswith("CONTINUITY NOTES:"):
            ep.continuity = line.split(":", 1)[1].strip()
        elif re.match(r"^SCENE \d+$", line):
            scene_no = int(line.split()[1])
        elif line.lower().startswith("scene heading:"):
            heading = line.split(":", 1)[1].strip()
        elif (sm := RE_SCENE_TIME.match(line)):
            t = to_sec(*sm.groups()[1:])
            if sm.group(1).lower() == "start":
                s_start = t
            else:
                ep.scenes.append(Scene(ep.episode, scene_no, heading, s_start, t))
        elif (lm := RE_LABEL.match(line)):
            pending_label, pending_time = lm.group(1).strip(), None
        elif (tm := RE_BLOCK_TIME.match(line)):
            g = tm.groups()
            pending_time = (to_sec(*g[:3]), to_sec(*g[3:]))
            ep.raw_blocks += 1
        elif pending_label and pending_time:
            kind, spk, delivery, offscreen = _parse_label(pending_label)
            text = line.strip('"').strip("“”") if kind == "dialogue" else line
            ep.events.append(Event(ep.episode, scene_no, idx, kind, spk, delivery,
                                   offscreen, text, *pending_time))
            idx += 1
            pending_label = pending_time = None

    ep.duration = max([s.end for s in ep.scenes] + [e.end for e in ep.events] + [0.0])
    return ep


def normalize_speakers(episodes: list[Episode], aliases: dict[str, str] | None = None) -> dict[str, str]:
    """Map raw speaker labels to a canonical name.

    Generic rule: drop an honorific prefix (LADY/LORD/THE/SIR) only when the bare
    name is itself a speaker in the show (LADY MARY -> MARY when MARY also speaks,
    but LADY GREY stays if GREY never speaks alone). Optional per-show aliases override.
    """
    names = {e.speaker for ep in episodes for e in ep.events if e.speaker}
    mapping = {}
    for n in names:
        canon = n
        for pre in ("LADY ", "LORD ", "THE ", "SIR "):
            if n.startswith(pre) and n[len(pre):] in names:
                canon = n[len(pre):]
        mapping[n] = canon
    if aliases:
        mapping.update({k.upper(): v.upper() for k, v in aliases.items()})
    return mapping


def load_pack(folder: Path, aliases: dict[str, str] | None = None) -> list[Episode]:
    files = sorted(p for p in Path(folder).rglob("*.txt"))
    eps = sorted((parse_file(p) for p in files), key=lambda e: e.episode)
    mapping = normalize_speakers(eps, aliases)
    for ep in eps:
        for ev in ep.events:
            if ev.speaker:
                ev.speaker = mapping.get(ev.speaker, ev.speaker)
    return eps


def validate(eps: list[Episode], tol: float = 0.6) -> list[str]:
    """Integrity checks. Returns a list of human-readable problems (empty = clean)."""
    issues = []
    nums = [e.episode for e in eps]
    if len(set(nums)) != len(nums):
        issues.append(f"duplicate episode numbers: {nums}")
    for ep in eps:
        if not ep.events:
            issues.append(f"ep{ep.episode}: no events parsed")
            continue
        if ep.raw_blocks != len(ep.events):
            issues.append(f"ep{ep.episode}: {ep.raw_blocks} timed blocks in file, {len(ep.events)} parsed")
        if not ep.scenes:
            issues.append(f"ep{ep.episode}: no scenes parsed")
        if ep.scenes and abs(ep.scenes[-1].end - max(e.end for e in ep.events)) > 2.0:
            issues.append(f"ep{ep.episode}: last scene ends {fmt(ep.scenes[-1].end)}, "
                          f"last event ends {fmt(max(e.end for e in ep.events))}")
        prev_end = 0.0
        for s in ep.scenes:
            if s.end <= s.start:
                issues.append(f"ep{ep.episode} sc{s.scene}: end<=start")
            if abs(s.start - prev_end) > tol:
                issues.append(f"ep{ep.episode} sc{s.scene}: gap/overlap {prev_end:.2f}->{s.start:.2f}")
            prev_end = s.end
        bounds = {s.scene: s for s in ep.scenes}
        for ev in ep.events:
            if ev.end < ev.start:
                issues.append(f"ep{ep.episode} #{ev.idx}: end<start")
            sc = bounds.get(ev.scene)
            if sc and (ev.start < sc.start - tol or ev.end > sc.end + tol):
                issues.append(f"ep{ep.episode} #{ev.idx} ({fmt(ev.start)}-{fmt(ev.end)}) "
                              f"outside scene {ev.scene} [{fmt(sc.start)}-{fmt(sc.end)}]")
    return issues


def write_tables(eps: list[Episode], out: Path) -> None:
    import csv, json
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(eps[0].events[0]).keys()))
        w.writeheader()
        for ep in eps:
            for ev in ep.events:
                w.writerow(asdict(ev))
    meta = [{"episode": e.episode, "file": e.file, "duration": e.duration,
             "logline": e.logline, "continuity": e.continuity,
             "scenes": [asdict(s) for s in e.scenes]} for e in eps]
    (out / "episodes.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import sys
    folder = Path(sys.argv[1])
    eps = load_pack(folder)
    problems = validate(eps)
    write_tables(eps, Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/parsed"))
    print(f"episodes={len(eps)} events={sum(len(e.events) for e in eps)} "
          f"scenes={sum(len(e.scenes) for e in eps)} problems={len(problems)}")
    for p in problems[:40]:
        print("  -", p)
