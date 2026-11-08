# Promo Finder — Kuku AI Generalist assignment (Problem 1)

By **Sahaj Yadav**.

Reads a pack of timestamped microdrama screenplays and returns ranked 20–60s promo cuts:
episode + start/end of every segment, the hook, the cliffhanger, why it works, and why it's ranked where it is.

## Run

```bash
pip install requests            # only dependency (pytest optional)
# put keys in a .env file next to this README (git-ignored), one per line:
#   GEMINI_API_KEY=...          free: https://aistudio.google.com/apikey
#   GROQ_API_KEY=...            free: https://console.groq.com/keys
python find_promos.py --input path/to/episodes/ --model gemini:gemini-3.8-flash \
    --judge-model groq:openai/gpt-oss-120b --out outputs/run
```

Swap the model with one flag; nothing else changes. Give a **comma list** to fall back automatically
when a free quota runs out or a model is retired; each promo records which model picked it:

```bash
python find_promos.py --input data/ --out outputs/run \
    --model "groq:openai/gpt-oss-120b,gemini:gemini-3.6-flash,gemini:gemini-3.5-flash" \
    --judge-model gemini:gemini-3-flash-preview
```

Free tiers are tight (Gemini: 20 requests/day per model; Groq: 200k tokens/day per model). A run
that hits a limit says which episodes are missing; re-run the same command and finished calls come
from the cache. Model names change often (`gemini-2.5-flash` and `llama-3.3-70b-versatile` were
retired while this was built), so check your provider's model list if you get a 404.

| `--model` | Needs |
|---|---|
| `gemini:gemini-3.8-flash` | `GEMINI_API_KEY` (free tier) |
| `groq:openai/gpt-oss-120b` | `GROQ_API_KEY` (free tier) |
| `ollama:qwen2.5:7b` | local [Ollama](https://ollama.com), no key |
| `openai:<model>` | `OPENAI_API_KEY`; set `OPENAI_BASE_URL` for any OpenAI-compatible API |
| `anthropic:<model>` | `ANTHROPIC_API_KEY` |

Useful flags: `--episodes 1-5` (quick trial), `--judge-model` (judge with a different model than the proposer),
`--baseline` (no model at all), `--workers 1` (if a free tier rate-limits you). LLM responses are cached in `.cache/`,
so re-running is free and reproducible; `--no-cache` forces fresh calls.

## Outputs (`--out` folder)

- `promos.csv` / `promos.md`: the ranked list, ready for an editor to review
- `promos.json`: everything, including the exact transcript a viewer would see for each cut
- `rejected.json`: picks the rule checker threw out, with the reason
- `bible.json`, `moments.json`, `logs.json`, `run_meta.json`: show summary, raw model replies, token usage, cost

## How it works

```
screenplays ─► parse (code) ─► show bible (LLM, 1 call)
                    │
                    ▼
     per episode: propose picks by beat ID (LLM) ─► measure + rule-check (code) ─► 1 repair round (LLM)
                    │
                    ▼
     cross-episode montage from standout moments (LLM, 1 call)
                    │
                    ▼
     judge each cut as a cold viewer, transcript only (LLM) ─► rank + spread across episodes (code)
```

- **The model never writes a timestamp.** It names beat IDs (`E07-031`); code converts them to times, snaps cuts to camera-shot
  boundaries without slicing a line, and enforces 20–60s, a hook in the first 3s, and a cliffhanger at the cut.
- **The model's quotes aren't trusted.** Hook and cliffhanger text is re-read from the script.
- **The judge sees only the clip**, not the show, so "makes sense on its own" is tested the way a stranger would experience it.

## Quality checks

```bash
python -m pytest -q                                                # parser + rule tests
python evaluate.py --run outputs/run --gold gold.csv               # recall against hand-picked promos
python evaluate.py --run outputs/run --compare outputs/run_b       # stability across runs / models
python find_promos.py --input ... --baseline --judge-model gemini:gemini-2.5-flash --out outputs/baseline
python evaluate.py --run outputs/run --compare outputs/baseline    # LLM vs keyword heuristic
```

The input pack is Kuku's confidential content and is not in this repo (`data/` is git-ignored).
