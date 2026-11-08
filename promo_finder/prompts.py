"""All prompts in one place, so they can be tuned without touching pipeline code."""

PROMO_DEFINITION = """\
A PROMO is a 20-60 second clip cut from the show and posted on TikTok/Instagram/YouTube
Shorts to make a stranger who is scrolling stop, watch, and download the app to see what
happens next. A great promo:
1. HOOK: opens on something that stops the scroll in the first 3 seconds - a slap, a
   shocking line, a public humiliation, a reveal, a threat. Never open on calm setup.
2. CLIFFHANGER: cuts at an unresolved moment - a question hanging, a reveal about to
   land, a confrontation mid-escalation. Never end on a resolution or a goodbye.
3. STANDS ALONE: a viewer who has never seen the show can follow who is wronging whom
   from what is said and shown inside the clip.
It is usually one continuous stretch, but may join 2-3 moments (from one episode or
several) when that makes a stronger hook-to-cliffhanger arc."""

BIBLE_SYSTEM = "You are a story editor at a microdrama studio. Reply with JSON only."

BIBLE_USER = """\
TASK: SHOW BIBLE
Below are the loglines and continuity notes for every episode of one show.
Write a compact bible a promo editor can use.

Return JSON:
{{"premise": "<2 sentences, what a new viewer is promised>",
  "characters": [{{"name": "<canonical>", "aliases": ["<other labels used in scripts>"],
                  "who": "<role in <=12 words>"}}],
  "reveals": [{{"episode": <int>, "reveal": "<major twist that would spoil later
               episodes if shown in an ad; skip the core premise in the show title>"}}]}}

SHOW TITLE: {title}

{episodes}"""

PROPOSE_SYSTEM = f"""\
You are the senior promo editor for a vertical microdrama app (like ReelShort/DramaBox).
You read screenplays and pick the moments that will perform as paid social ads.

{PROMO_DEFINITION}

The script lists every beat with an ID like E07-031 and its timestamps. "cam" beats are
camera descriptions; they often carry the visual hook (a slap, a ring, a face), so use them.

Rules for your picks:
- Each segment is a continuous range of beats inside ONE episode: first_beat..last_beat.
- Total length of all segments: 20-60 seconds (aim for 25-45). Check the timestamps.
- The hook beat must begin within 3 seconds of the first segment's start: start ON it,
  or one beat before it at most.
- The last segment must end on the cliffhanger beat. Do not include the payoff.
- Prefer picks a stranger understands; if a viewer would need prior context, say what.
- Quality over quantity. Give 1-3 promos for this episode; give 0 if nothing is strong.
- Also list up to 3 standout single moments (the strongest hook or cliffhanger beats)
  so they can be combined with moments from other episodes.
- Quote the hook and cliffhanger text exactly as written in the script.
Reply with JSON only."""

PROPOSE_USER = """\
TASK: PROPOSE
SHOW: {title}
PREMISE: {premise}
MAIN CHARACTERS: {characters}
This is episode {episode} of {n_episodes}.

SCRIPT:
{script}

Return JSON:
{{"promos": [{{"title": "<3-6 word working title>",
              "segments": [{{"first_beat": "E..-...", "last_beat": "E..-..."}}],
              "hook_beat": "E..-...", "hook": "<exact text>",
              "cliffhanger_beat": "E..-...", "cliffhanger": "<exact text>",
              "why": "<one line on why a stranger stops and needs the next episode>",
              "context_needed": "<what a new viewer must already know, or empty>"}}],
  "moments": [{{"beat": "E..-...", "type": "hook|cliffhanger", "text": "<exact text>"}}]}}"""

REPAIR_USER = """\
{original}

Your previous picks broke these rules after the cuts were measured on the real timeline:
{problems}

Return the corrected JSON (same schema). Fix or drop the broken promos; keep good ones as they were."""

MONTAGE_USER = """\
TASK: MONTAGE
SHOW: {title}
PREMISE: {premise}

Below are standout moments from across the show, each with the beats around it.
Build up to {k} promos that JOIN 2-3 moments from DIFFERENT episodes into one
hook -> escalation -> cliffhanger arc (e.g. a public humiliation in an early
episode, then a reveal of who someone really is, cut before anyone reacts).
Each segment must be a continuous range inside one episode, at least 4 seconds.
Total 20-60 seconds. Segments play in the order you list them.

MOMENTS:
{moments}

Return JSON:
{{"promos": [{{"title": "...", "segments": [{{"first_beat": "E..-...", "last_beat": "E..-..."}}],
              "hook_beat": "E..-...", "hook": "<exact text>",
              "cliffhanger_beat": "E..-...", "cliffhanger": "<exact text>",
              "why": "<one line>", "context_needed": ""}}]}}"""

JUDGE_SYSTEM = """\
You are a skeptical growth marketer reviewing ad cuts for a microdrama app.
You see ONLY what a stranger scrolling TikTok would see and hear in each clip -
nothing about the rest of the show. Judge strictly; most cuts should NOT get 5s.
Reply with JSON only."""

JUDGE_USER = """\
TASK: JUDGE
""" + PROMO_DEFINITION + """

Score each promo 1-5 on:
- hook: do the first 3 seconds (lines marked >>) stop the scroll?
- cliffhanger: does the final beat leave an urgent unanswered question? (5 = I must know)
- standalone: can a stranger follow who is who and what is at stake from the clip alone?
- spoiler: 0 = none, 1 = mild, 2 = gives away one of these later-season reveals:
{reveals}

{promos}

Return JSON: {{"scores": [{{"id": <promo number>, "hook": n, "cliffhanger": n,
  "standalone": n, "spoiler": n, "note": "<the single biggest weakness, <=15 words>"}}]}}"""
