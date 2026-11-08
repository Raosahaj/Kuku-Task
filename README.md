# Promo Finder

**By Sahaj Yadav** · Kuku AI Generalist assignment, Problem 1

Reads the timestamped screenplays of a microdrama and returns a **ranked list of 20–60 second promo
cuts**: for each one, the episode and exact start/end of every segment, the hook (first 3 seconds),
the cliffhanger (where it cuts), one line on why it works, and why it sits at that rank.
An editor pastes the timestamps into their editing software instead of watching every episode.

---

## Try it in 10 seconds (no API key, no data)

The repo ships a short **made-up** two-episode show in [`examples/sample_show/`](examples/sample_show/).
`mock` is an offline stand-in model, so this only proves the pipeline runs end to end:

```bash
pip install requests
python find_promos.py --input examples/sample_show --model mock --out outputs/demo
```

To see what a **real model** makes of the same sample, open
[`examples/sample_output/promos.md`](examples/sample_output/promos.md). Its #1 pick:

```
#1 · Engagement Party Bakery Betrayal · score 80.0
Cut:          E01 00:08.50-00:48.00 (40s)
Hook:         "This is a sale contract. For my mother's bakery."
Cliffhanger:  "Who are you?"
Why it works: Stops scrolling with a shocking betrayal at an engagement party and cuts
              right as an imposing stranger unmasks his authority.
```

## Run it on real screenplays

1. Put a show's episode files (`episode-001-....txt`, one per episode) in a folder, e.g. `data/`.
2. Create a `.env` file next to this README (it is git-ignored) with one or more free keys:
   ```
   GEMINI_API_KEY=...    # https://aistudio.google.com/apikey
   GROQ_API_KEY=...      # https://console.groq.com/keys
   ```
3. Run:
   ```bash
   python find_promos.py --input data --out outputs/run \
       --model "groq:openai/gpt-oss-120b,gemini:gemini-3.6-flash,gemini:gemini-3.5-flash" \
       --judge-model gemini:gemini-3-flash-preview
   ```
4. Open `outputs/run/promos.md` (readable) or `outputs/run/promos.csv` (for an editor).

Quick trial on a few episodes: add `--episodes 1-5`. No model at all: `--baseline`.

## What you get

| File | What it is |
|---|---|
| `promos.md` / `promos.csv` | The ranked promo list: cut times, hook, cliffhanger, why it works, why this rank |
| `promos.json` | Everything above, plus the exact transcript a viewer would see/hear in each cut and all scores |
| `rejected.json` | Picks the rule checker threw out, with the reason (e.g. "duration 61.2s outside 20–60s") |
| `logs.json` | Raw model replies per episode, including which model answered |
| `bible.json`, `moments.json` | Show summary and standout moments the models worked from |
| `run_meta.json` | Token usage, run time, and any episodes or promos that were skipped |

## How it works

```
screenplays ──► parse (code) ──► show summary (AI, 1 call)
                                      │
   for each episode:  AI proposes cuts by line ID ──► code measures + checks rules ──► 1 AI repair round
                                      │
                      cross-episode montage (AI, 1 call)
                                      │
   judge each cut as a stranger would (a different AI) ──► rank + spread across episodes (code)
```

Three rules keep the AI honest:

- **The AI never writes a timestamp.** It names line IDs like `E07-031`. Code converts them to times,
  snaps each cut to a camera-shot boundary without slicing a line of dialogue, and enforces the promo
  rules: 20–60 seconds, a hook in the first 3 seconds, the cliffhanger at the very end.
- **The AI's quotes aren't trusted.** Hook and cliffhanger text is re-read from the script.
- **The judge sees only the clip.** It gets the transcript of the cut, not the show, so "makes sense on
  its own" is tested the way a stranger scrolling TikTok would experience it.

Ranking = judge scores (hook 35%, cliffhanger 30%, stands-alone 25%) + a small boost for early
episodes − a spoiler penalty. Posting order then avoids repeating the same episode back to back.

## Choosing a model

Swap models with one flag; nothing else changes. A **comma-separated list** falls back automatically
when a free quota runs out or a model is retired.

| `--model` | Needs |
|---|---|
| `gemini:<model>` e.g. `gemini:gemini-3.6-flash` | `GEMINI_API_KEY` (free tier: 20 requests/day per model) |
| `groq:<model>` e.g. `groq:openai/gpt-oss-120b` | `GROQ_API_KEY` (free tier: 200k tokens/day per model) |
| `openai:<model>` | `OPENAI_API_KEY`; set `OPENAI_BASE_URL` for **any OpenAI-compatible API** |
| `anthropic:<model>` | `ANTHROPIC_API_KEY` |
| `ollama:<model>` | a local [Ollama](https://ollama.com) server, no key |
| `mock` | nothing (offline stand-in for tests and demos) |

Every model reply is cached in `.cache/`, so re-running is free and repeatable, and a run that hit a
quota only pays for what's missing. The run ends with a warning listing any episode or promo that was
skipped. Free-tier model names change often; if you get a 404, check your provider's model list.

## Repo layout

```
find_promos.py           command-line entry point: run the pipeline, write all outputs
evaluate.py              quality report on a finished run (rules, recall, comparison with another run)
promo_finder/
  parse.py               screenplay text → table of timed lines (pure code)
  candidates.py          line IDs → measured cuts, shot snapping, the hard rules, no-AI baseline
  pipeline.py            propose → repair → montage → judge → rank
  prompts.py             every prompt in one place, editable without touching code
  models.py              one adapter for all providers, fallback chains, caching, rate-limit handling
tests/                   rule and parser tests; test_sample.py runs anywhere, test_core.py needs the data pack
examples/sample_show/    a made-up two-episode show in the expected screenplay format
examples/sample_output/  real model output on that sample
```

## Checking quality

```bash
python -m pytest -q                                              # tests
python evaluate.py --run outputs/run --gold reference.csv        # recall vs hand-picked moments
python evaluate.py --run outputs/run --compare outputs/other     # agreement with another run/model
python find_promos.py --input data --baseline --out outputs/baseline   # no-AI yardstick
```

`reference.csv` columns: `episode,start,end,note` with times as `MM:SS.xx`. A reference moment counts as
found if a promo overlaps at least half of it.

## Known limitations

- Promos that run a second over 60s are dropped rather than trimmed.
- Quiet, low-conflict episodes still get a pick (the ranking sinks them, but it shouldn't propose them).
- Cross-episode montages often fail the rules, so most promos are one continuous stretch.
- Some dark scenes are refused by the judge model's safety filter; they are flagged "check ad-platform
  policy by hand" instead of being scored.

## Data

The show's screenplays are Kuku's confidential content and are **not** in this repo: `data/`,
`outputs/` and `.env` are git-ignored. Everything under `examples/` is made up.
