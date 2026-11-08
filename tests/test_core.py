"""Fast checks on the deterministic parts. Run: python -m pytest -q"""
from pathlib import Path

import pytest

from promo_finder.candidates import Promo, Segment, Show
from promo_finder.parse import load_pack, parse_file, validate
from promo_finder.pipeline import to_promo

DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(not any(DATA.rglob("*.txt")), reason="input pack not present")


@pytest.fixture(scope="module")
def show():
    return Show(load_pack(DATA))


def test_pack_parses_clean():
    eps = load_pack(DATA)
    assert len(eps) == 50 and validate(eps) == []
    assert all(e.raw_blocks == len(e.events) for e in eps)


def test_uppercase_scene_headers(tmp_path):
    f = tmp_path / "episode-001-x.txt"
    f.write_text("EPISODE NUMBER: Episode 1\nSCENE 01\nSCENE HEADING: INT. X\n"
                 "SCENE START: 00:00.00\nSCENE END: 00:10.00\n**JOHN (Calm):**\n"
                 "Start: (00:01.00) | End: (00:02.00)\n\"Hi.\"\n")
    ep = parse_file(f)
    assert len(ep.scenes) == 1 and ep.events[0].speaker == "JOHN" and ep.events[0].text == "Hi."


def test_rules_reject_bad_cuts(show):
    assert any("duration" in i for i in show.check(Promo([Segment(1, 0.0, 10.0)])))
    assert any("mid-line" in i for i in show.check(Promo([Segment(1, 16.0, 49.0)])))


def test_model_cannot_invent_beats(show):
    p = to_promo(show, {"segments": [{"first_beat": "E01-999", "last_beat": "E01-010"}]}, "t")
    assert p.issues and "invalid" in p.issues[0]


def test_hook_must_open_the_promo(show):
    d = {"segments": [{"first_beat": "E01-007", "last_beat": "E01-040"}],
         "hook_beat": "E01-030", "cliffhanger_beat": "E01-040"}
    assert any("hook beat" in i for i in to_promo(show, d, "t").issues)


def test_real_text_replaces_model_quotes(show):
    d = {"segments": [{"first_beat": "E01-006", "last_beat": "E01-040"}],
         "hook_beat": "E01-007", "hook": "something the model made up",
         "cliffhanger_beat": "E01-040"}
    p = to_promo(show, d, "t")
    assert p.hook == show.event("E01-007").text and p.scores.get("quote_mismatch")


class _ScriptedLLM:
    """Replays canned judge replies (or raises them) instead of calling a model."""
    def __init__(self, replies):
        self.replies = list(replies)

    def json(self, system, user):
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _two_promos():
    return [Promo([Segment(1, 0.0, 30.0)]), Promo([Segment(2, 0.0, 30.0)])]


def test_judge_accepts_bare_list_and_retries_empty_reply(show):
    from promo_finder.pipeline import judge
    ps = _two_promos()   # first reply is empty (as Gemini previews sometimes are), second is a bare list
    judge(_ScriptedLLM([ValueError("gemini returned no text"),
                        [{"id": 0, "hook": 4}, {"id": 1, "hook": 2}]]), show, ps, {})
    assert [p.scores.get("hook") for p in ps] == [4, 2]
    assert not any(p.scores.get("judge_error") for p in ps)


def test_unjudged_promo_is_flagged_not_silently_ranked(show):
    from promo_finder.pipeline import judge, rank, rank_reason
    ps = _two_promos()
    judge(_ScriptedLLM([ValueError("down"), ValueError("still down")]), show, ps, {})
    rank(ps, 50)
    assert all(rank_reason(p).startswith("NOT JUDGED") for p in ps)


def test_fallback_chain_moves_on_when_quota_runs_out(tmp_path, monkeypatch):
    import promo_finder.models as M
    monkeypatch.setattr(M, "CACHE", tmp_path)
    llm = M.make_llm("groq:primary,mock")

    def out_of_quota(system, user):
        raise M.QuotaExhausted("primary: provider quota exhausted")
    monkeypatch.setattr(llm.chain[0], "_call", out_of_quota, raising=False)

    llm.complete("sys", "TASK: JUDGE\nPROMO 0")
    assert llm.last_spec == "mock" and "groq:primary" in llm.exhausted
    assert [u["api_calls"] for u in llm.usages()] == [0, 1]


def test_safety_blocked_batch_is_judged_one_by_one(show):
    from promo_finder.pipeline import judge
    ps = _two_promos()
    blocked = ValueError("gemini returned no text (reason: {'blockReason': 'PROHIBITED_CONTENT'})")
    # batch blocked -> promo 0 judged alone fine -> promo 1 blocked alone
    judge(_ScriptedLLM([blocked, [{"id": 0, "hook": 4}], blocked]), show, ps, {})
    assert ps[0].scores.get("hook") == 4 and not ps[0].scores.get("judge_error")
    assert "safety filter" in ps[1].scores["judge_error"]
