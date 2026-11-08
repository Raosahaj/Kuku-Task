"""Checks that run anywhere, on the made-up sample show in examples/ (no data pack, no API key)."""
from pathlib import Path

from promo_finder.candidates import MAX_LEN, MIN_LEN, Show
from promo_finder.models import make_llm
from promo_finder.parse import load_pack, validate
from promo_finder.pipeline import run

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample_show"


def test_sample_parses_clean():
    eps = load_pack(SAMPLE)
    assert [e.episode for e in eps] == [1, 2]
    assert validate(eps) == []


def test_pipeline_end_to_end_offline(tmp_path, monkeypatch):
    import promo_finder.models as M
    monkeypatch.setattr(M, "CACHE", tmp_path)
    eps = load_pack(SAMPLE)
    show = Show(eps)
    llm = make_llm("mock")
    res = run(show, eps, "sample", llm, llm, workers=1, log=lambda *_: None)
    assert res["ranked"], "the offline stand-in model should still yield valid promos"
    for p in res["ranked"]:
        assert MIN_LEN - 0.5 <= p.duration <= MAX_LEN + 0.5 and not p.issues
        assert p.scores["rank"] >= 1
