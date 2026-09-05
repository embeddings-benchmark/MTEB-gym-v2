"""
Smoke test: full pipeline on a tiny synthetic corpus with the MockClient.
No network, no GPU, no API key. Run: PYTHONPATH=. python3 tests/test_pipeline.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gym.clients import MockClient
from gym.config import GymConfig
from gym.judge import Judge, _parse_response
from gym.query_generator import Query, QueryGenerator, _extract_json
from gym.retrieval_harness import Retrieved
from gym.scoring import rate, format_leaderboard
from gym.validate import correlate


def make_corpus(n=40):
    topics = ["heart disease and statins", "vitamin D and asthma",
              "gut microbiome and fiber", "telomeres and stress"]
    corpus = {}
    for i in range(n):
        t = topics[i % len(topics)]
        corpus[f"D{i}"] = f"Document {i} about {t}. " + ("clinical evidence " * (i % 5 + 1))
    return corpus


def test_json_extraction():
    assert _extract_json('```json\n{"winner": "A"}\n```')["winner"] == "A"
    assert _extract_json('blah {"score": 4} trailing')["score"] == 4
    assert _extract_json("not json") == {}
    # reasoning trace before the answer: take the LAST object, not the first
    assert _extract_json('Let me think {"winner": "B"} ... final {"winner": "A"}')["winner"] == "A"
    # braces inside a string value must not truncate the parse
    assert _extract_json('{"reasoning": "set {x} wins", "winner": "B"}')["winner"] == "B"
    print("  json extraction ok")


def test_query_gen_and_filter():
    cfg = GymConfig(n_queries=8, gen_overshoot=2.0, filter_queries=True,
                    output_dir="results/_test")
    corpus = make_corpus()
    gen = QueryGenerator(MockClient(), cfg)
    raw = gen.generate(corpus)
    assert len(raw) >= 8, f"only generated {len(raw)}"
    kept = gen.filter(raw)
    assert len(kept) <= 8
    assert all(q.quality is not None for q in kept)
    print(f"  query gen+filter ok ({len(raw)} -> {len(kept)} kept)")


def test_search_retriever():
    try:
        import bm25s  # noqa: F401
        import mteb
    except ImportError:
        print("  search retriever: mteb/bm25s absent, skipped")
        return
    from gym.baselines import MTEBSearchRetriever
    meta = mteb.get_tasks(tasks=["NFCorpus"])[0].metadata
    corpus = make_corpus()
    queries = [Query("q0", "statins and heart disease evidence", ["D0"])]
    res = MTEBSearchRetriever("bm25", meta, top_k=5).retrieve(corpus, queries)
    assert len(res) == 1 and len(res[0].doc_ids) == 5
    assert all(d in corpus for d in res[0].doc_ids)
    print("  search retriever (mteb bm25) ok")


def test_env_knobs_rejected():
    import os
    from gym.gym import Gym
    os.environ["GYM_MAX_CORPUS_DOCS"] = "5"
    try:
        Gym(GymConfig(judge="mock"))
        raise AssertionError("a stale GYM_* variable must be rejected")
    except RuntimeError as e:
        assert "GYM_MAX_CORPUS_DOCS" in str(e)
    finally:
        del os.environ["GYM_MAX_CORPUS_DOCS"]
    print("  stale env knobs rejected ok")


def test_judge_and_scoring():
    # Build fake retrievals for 3 models over 30 queries.
    queries = [Query(f"q{i}", f"query {i} about statins", ["D0"]) for i in range(30)]

    def fake_retr(seed):
        out = []
        for q in queries:
            docs = [f"result {seed}-{q.qid}-{j}" for j in range(5)]
            out.append(Retrieved(q.qid, q.text, [f"D{j}" for j in range(5)], docs))
        return out

    judge = Judge(MockClient(seed=1), flip_positions=True)
    v_ab = judge.judge_all(fake_retr("a"), fake_retr("b"), "model_a", "model_b")
    v_ac = judge.judge_all(fake_retr("a"), fake_retr("c"), "model_a", "model_c")
    v_bc = judge.judge_all(fake_retr("b"), fake_retr("c"), "model_b", "model_c")
    verdicts = v_ab + v_ac + v_bc

    assert all(0.0 <= v.score_a <= 1.0 for v in verdicts)
    assert judge.a_first_rate is not None
    print(f"  judge ok (a_first_rate={judge.a_first_rate:.2f}, {len(verdicts)} verdicts)")

    ratings = rate(verdicts, bootstrap=200)
    assert len(ratings) == 3
    assert ratings[0].rating >= ratings[-1].rating
    assert ratings[0].ci >= 0
    print("  bradley-terry scoring ok")
    print(format_leaderboard(ratings))
    return ratings


def test_correlation(ratings):
    gym_ratings = {m.name: m.rating for m in ratings}
    # perfectly aligned ground truth -> rho should be 1.0
    truth = {m.name: float(len(ratings) - i) for i, m in enumerate(ratings)}
    res = correlate(gym_ratings, truth, bootstrap=200)
    assert abs(res["spearman_rho"] - 1.0) < 1e-9, res
    print(f"  correlation ok (rho={res['spearman_rho']:.2f})")


def test_exact_permutation_p():
    # identical rankings over 5 models -> p = 2/5! exactly (only the identity
    # and the full reversal reach |rho| = 1)
    g = {f"m{i}": float(i) for i in range(5)}
    res = correlate(g, dict(g), bootstrap=50)
    assert res["spearman_p_exact"] is not None
    assert abs(res["spearman_p_exact"] - 2 / 120) < 1e-9, res["spearman_p_exact"]
    print(f"  exact permutation p ok (p={res['spearman_p_exact']:.4f})")


def _fake_retrievals(seed, queries, k=5):
    return [Retrieved(q.qid, q.text, [f"D{seed}{j}" for j in range(k)],
                      [f"result {seed}-{q.qid}-{j}" for j in range(k)])
            for q in queries]


def test_parallel_judging_matches_sequential():
    queries = [Query(f"q{i}", f"query {i} about statins", ["D0"]) for i in range(30)]
    ra, rb = _fake_retrievals("a", queries), _fake_retrievals("b", queries)

    seq = Judge(MockClient(seed=1), workers=1).judge_all(ra, rb, "m_a", "m_b")
    par = Judge(MockClient(seed=1), workers=8).judge_all(ra, rb, "m_a", "m_b")

    assert [v.qid for v in par] == [v.qid for v in seq]
    assert [v.score_a for v in par] == [v.score_a for v in seq]
    print(f"  parallel judging deterministic ok ({len(par)} verdicts)")


def test_parallel_query_gen_matches_sequential():
    # Worker count must never change the generated query set: texts, seed
    # docs, and qids must be identical, in the same order, at workers=1 and 4.
    corpus = make_corpus()

    def gen_with(workers, client):
        cfg = GymConfig(n_queries=12, filter_queries=False,
                        gen_workers=workers, output_dir="results/_test")
        return QueryGenerator(client, cfg).generate(corpus)

    seq = gen_with(1, MockClient())
    par = gen_with(4, MockClient())
    assert [q.qid for q in par] == [q.qid for q in seq]
    assert [q.text for q in par] == [q.text for q in seq]
    assert [q.seed_doc_ids for q in par] == [q.seed_doc_ids for q in seq]

    # A client that deterministically fails ~1/3 of calls (by content hash,
    # so thread scheduling can't change which calls fail) exercises the
    # wave-refill retry path; results must still match exactly.
    class FlakyMock(MockClient):
        def chat(self, messages, temperature=0.0):
            prompt = " ".join(m.get("content", "") for m in messages)
            if self._hash(prompt) % 3 == 0:
                return "no json here"
            return super().chat(messages, temperature)

    seq_f = gen_with(1, FlakyMock())
    par_f = gen_with(4, FlakyMock())
    assert len(seq_f) == 12, f"flaky retries should still reach target, got {len(seq_f)}"
    assert [(q.qid, q.text, tuple(q.seed_doc_ids)) for q in par_f] == \
           [(q.qid, q.text, tuple(q.seed_doc_ids)) for q in seq_f]
    print(f"  parallel query generation deterministic ok "
          f"({len(par)} clean, {len(par_f)} with retries)")


def test_identical_results_short_circuit():
    class ExplodingClient:
        def chat(self, messages, temperature=0.0):
            raise AssertionError("judge must not be called for identical result sets")

    queries = [Query("q0", "some query", ["D0"])]
    ra = _fake_retrievals("same", queries)
    rb = _fake_retrievals("same", queries)
    verdicts = Judge(ExplodingClient()).judge_all(ra, rb, "m_a", "m_b")
    assert verdicts[0].score_a == 0.5
    assert verdicts[0].raw == ["identical"]
    print("  identical-results short-circuit ok")


def test_parse_failure_counter():
    class GarbageClient:
        def chat(self, messages, temperature=0.0):
            return "I refuse to answer in the requested format."

    queries = [Query(f"q{i}", f"query {i}", ["D0"]) for i in range(5)]
    ra, rb = _fake_retrievals("a", queries), _fake_retrievals("b", queries)
    judge = Judge(GarbageClient())
    verdicts = judge.judge_all(ra, rb, "m_a", "m_b")
    assert all(v.score_a == 0.5 for v in verdicts)
    assert judge.parse_failure_rate == 1.0
    print("  parse-failure counter ok (rate=1.0 on garbage judge)")


def test_winless_model_ranks_last():
    from gym.judge import Verdict

    def v(qid, a, b, score):
        return Verdict(qid=qid, query="q", model_a=a, model_b=b, score_a=score)

    verdicts = []
    for i in range(20):
        verdicts.append(v(f"q{i}", "winner", "mid", 0.75))
        verdicts.append(v(f"q{i}", "mid", "loser", 1.0))
        verdicts.append(v(f"q{i}", "winner", "loser", 1.0))
    ratings = rate(verdicts, bootstrap=100)
    assert ratings[-1].name == "loser"
    assert ratings[-1].rating < ratings[1].rating - 50, \
        "a model that loses every verdict must sink, not sit mid-table"
    assert all(r.ci >= 0 for r in ratings)
    print("  winless-model + cluster bootstrap ok")


def test_corpus_fingerprint_content_sensitive():
    from gym.retrieval_harness import _corpus_fingerprint

    base = _corpus_fingerprint({"D0": "alpha", "D1": "beta"})
    assert _corpus_fingerprint({"D0": "alpha", "D1": "beta"}) == base, "deterministic"
    assert _corpus_fingerprint({"D0": "alpha", "D1": "CHANGED"}) != base, \
        "a doc text change must invalidate the cache (length+ids were not enough)"
    assert _corpus_fingerprint({"D0": "alpha", "D9": "beta"}) != base, \
        "a doc id change must invalidate the cache"
    print("  corpus fingerprint content sensitivity ok")


def test_matchup_resume_from_jsonl(tmp_dir="results/_test_resume"):
    import shutil
    from gym.gym import Gym

    shutil.rmtree(tmp_dir, ignore_errors=True)
    calls = {"n": 0}

    class CountingClient(MockClient):
        def chat(self, messages, temperature=0.0):
            calls["n"] += 1
            return super().chat(messages, temperature)

    queries = [Query(f"q{i}", f"query {i} about statins", ["D0"]) for i in range(6)]
    ra, rb = _fake_retrievals("a", queries), _fake_retrievals("b", queries)

    cfg = GymConfig(output_dir=tmp_dir, judge_workers=1)
    g = Gym(cfg, judge_client=CountingClient(seed=1))
    g._retr_cache = {"m_a": ra, "m_b": rb}
    full = g.matchup("m_a", "m_b", {}, queries)
    assert len(full) == 6
    first_run_calls = calls["n"]

    # Simulate a crash mid-pair: drop the finalized file, truncate the JSONL to
    # two verdicts, rerun. Only the four missing queries may hit the client.
    final = g._matchup_path("m_a", "m_b", queries)
    jsonl = final.with_suffix(".jsonl")
    jsonl.write_text("\n".join(jsonl.read_text().splitlines()[:2]) + "\n")
    final.unlink()

    calls["n"] = 0
    g2 = Gym(cfg, judge_client=CountingClient(seed=1))
    g2._retr_cache = {"m_a": ra, "m_b": rb}
    resumed = g2.matchup("m_a", "m_b", {}, queries)
    assert len(resumed) == 6
    assert [v.qid for v in resumed] == [q.qid for q in queries]
    assert calls["n"] == first_run_calls * 4 // 6, \
        f"resume must only judge the 4 missing queries, made {calls['n']} calls"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("  matchup JSONL resume ok (resume judged only the missing queries)")


def test_repro_helpers():
    from gym.repro import canonical_config, config_hash, git_revision

    a = {"task": "NFCorpus", "n_queries": 300, "seed": 0}
    b = {"seed": 0, "n_queries": 300, "task": "NFCorpus"}

    assert canonical_config(a) == canonical_config(b)
    assert config_hash(a) == config_hash(b)
    assert len(config_hash(a)) == 8

    rev = git_revision()
    assert rev is None or len(rev) == 40
    print("  reproducibility helpers ok")



def test_result_helpers():
    import json
    import tempfile

    from gym.results import (
        experiment_config_hash,
        result_directory,
        runtime_versions,
    )

    cfg = {
        "task_name": "NFCorpus",
        "n_queries": 300,
        "top_k": 10,
        "seed": 0,
    }

    versions = runtime_versions()
    assert "mteb_version" in versions
    assert "gym_version" in versions
    assert "gym_revision" in versions

    # Legacy/default synthetic arm keeps the established path.
    synthetic_path = result_directory(
        "results",
        "Qwen/Qwen3.6-27B",
        "MiniMaxAI/MiniMax-M2.7",
        cfg,
    )
    assert str(synthetic_path) == (
        "results/Qwen-Qwen3.6-27B__MiniMaxAI-MiniMax-M2.7/45a0e48a"
    )

    # Human-query runs get an explicit human-queries arm.
    human_cfg = {**cfg, "arm": "human"}
    human_path = result_directory(
        "results",
        "Qwen/Qwen3.6-27B",
        None,
        human_cfg,
    )
    assert human_path.parent.name == "Qwen-Qwen3.6-27B__human-queries"
    assert human_path.name == experiment_config_hash(human_cfg)

    # Serializing config_hash must not alter the underlying experiment hash.
    hashed_cfg = dict(human_cfg)
    hashed_cfg["config_hash"] = experiment_config_hash(hashed_cfg)
    assert experiment_config_hash(hashed_cfg) == hashed_cfg["config_hash"]

    # A task result lands at <arm>/<config-hash>/<task>.json.
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = result_directory(
            tmp,
            "Qwen/Qwen3.6-27B",
            "MiniMaxAI/MiniMax-M2.7",
            cfg,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        result_path = out_dir / "NFCorpus.json"
        result_path.write_text(json.dumps({
            "task_name": "NFCorpus",
            "config": cfg,
        }))

        assert result_path.exists()
        assert json.loads(result_path.read_text())["task_name"] == "NFCorpus"
        assert result_path.parent.name == experiment_config_hash(cfg)

    print("  standardized result helpers ok")



def test_unified_gym_api():
    import tempfile
    from pathlib import Path

    from gym import Gym, GymConfig
    from gym.clients import MockClient

    with tempfile.TemporaryDirectory() as tmp:
        cfg = GymConfig(
            judge="mock",
            models=["model_a", "model_b"],
            output_dir=Path(tmp),
        )
        gym = Gym(cfg)

        assert isinstance(gym.judge_client, MockClient)
        assert gym.cfg.models == ["model_a", "model_b"]

    print("  unified Gym API ok")



def test_rank_agreement_api():
    import json
    import tempfile
    from pathlib import Path

    import gym.validate as gv

    original_fetch = gv.fetch_truth
    try:
        gv.fetch_truth = lambda models, task="NFCorpus", split="test": {
            "model_a": 30.0,
            "model_b": 20.0,
            "model_c": 10.0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "NFCorpus.json"
            result_path.write_text(json.dumps({
                "task_name": "NFCorpus",
                "ratings": [
                    {"model": "model_a", "rating": 1100},
                    {"model": "model_b", "rating": 1000},
                    {"model": "model_c", "rating": 900},
                ],
                "scores": {
                    "test": [{
                        "main_score": None,
                        "spearman": None,
                        "spearman_ci_low": None,
                        "spearman_ci_high": None,
                        "spearman_top10": None,
                        "kendall": None,
                        "kendall_ap": None,
                        "p_permutation": None,
                        "n_models": 3,
                    }]
                },
            }))

            out = gv.rank_agreement(result_path, bootstrap=100, seed=0)
            updated = json.loads(result_path.read_text())
            row = updated["scores"]["test"][0]

            assert row["spearman"] == 1.0
            assert row["kendall"] == 1.0
            assert row["n_models"] == 3
            assert row["p_permutation"] is not None
            assert str(result_path) in out
    finally:
        gv.fetch_truth = original_fetch

    print("  rank agreement API ok")



def test_rank_agreement_mteb_fallback():
    import sys
    import types

    import gym.validate as gv

    calls = []

    class FakeCache:
        def download_from_remote(self):
            calls.append("download")

    class FakeTaskResult:
        task_name = "NFCorpus"
        scores = {"test": [{"ndcg_at_10": 0.42}]}

        def get_score(self, splits=None, getter=None):
            assert splits == ["test"]
            return getter(self.scores["test"][0])

    class FakeModelResult:
        task_results = [FakeTaskResult()]

    def fake_get_tasks(tasks):
        assert tasks == ["NFCorpus"]
        return [object()]

    def fake_get_model_meta(model):
        assert model == "org/model"
        calls.append("meta")
        return ("meta", model)

    def fake_get_model(model):
        raise AssertionError("get_model fallback should not be needed")

    def fake_evaluate(model, task, **kwargs):
        assert model == ("meta", "org/model")
        assert kwargs["overwrite_strategy"] == "only-missing"
        assert kwargs["show_progress_bar"] is False
        assert isinstance(kwargs["cache"], FakeCache)
        calls.append("evaluate")
        return FakeModelResult()

    fake_mteb = types.SimpleNamespace(
        ResultCache=FakeCache,
        get_tasks=fake_get_tasks,
        get_model_meta=fake_get_model_meta,
        get_model=fake_get_model,
        evaluate=fake_evaluate,
    )

    old_mteb = sys.modules.get("mteb")
    sys.modules["mteb"] = fake_mteb
    try:
        out = gv.fetch_truth(["org/model"], task="NFCorpus", split="test")
        assert out == {"org/model": 42.0}
        assert calls == ["download", "meta", "evaluate"]
    finally:
        if old_mteb is None:
            sys.modules.pop("mteb", None)
        else:
            sys.modules["mteb"] = old_mteb

    print("  rank agreement MTEB fallback ok")

def test_corpus_control_resolution():
    # corpus controls are part of cache identity; uncapped runs keep their key
    from gym.gym import Gym
    gym = object.__new__(Gym)
    gym.gen_client = type("Client", (), {"model": "generator"})()
    gym.cfg = GymConfig(corpus_cap=456, inject_qrels_docs="dev")
    capped = gym._query_config_hash()
    gym.cfg.inject_qrels_docs = "test"
    assert gym._query_config_hash() != capped
    gym.cfg = GymConfig()
    assert gym._query_config_hash() not in (capped,)
    print("  corpus control resolution ok")


def test_judge_instruction_resolution():
    from gym.config import GymConfig
    from gym.judge import _JUDGE_SYSTEM, judge_system
    from gym.results import judge_instruction_metadata as meta
    assert judge_system() == _JUDGE_SYSTEM  # generic prompt unchanged: caches rest on this
    assert meta(GymConfig(judge_instruction=None))["instruction"] is None
    assert meta(GymConfig(judge_instruction="Find docs that refute")) == {
        "instruction": "Find docs that refute", "instruction_source": "config:judge_instruction"}
    assert meta(GymConfig(corpus_path="/x"))["instruction"] is None  # no task to look up
    try:
        import mteb
    except ImportError:
        assert meta(GymConfig(task_name="ArguAna"))["instruction"] is None
        return
    from gym.judge import task_prompt
    assert task_prompt("BrightBiologyRetrieval") is None  # encoder prefix, not a criterion
    m = meta(GymConfig(task_name="ArguAna"))
    assert m["instruction_source"] == "mteb:task_prompt" and "refute the claim" in m["instruction"]

def test_top10_and_tau_ap():
    import numpy as np
    from gym.validate import correlate, _tau_ap
    rng = np.random.default_rng(0)
    truth = {f"m{i}": float(i) for i in range(25)}
    # strong models (15..24) shuffled among themselves, weak ordered: aggregate
    # rho stays high, top-10 agreement collapses
    top = list(range(15, 25)); rng.shuffle(top)
    gym_scores = {f"m{i}": float(v) for i, v in zip(range(15, 25), top)}
    gym_scores.update({f"m{i}": float(i) for i in range(15)})
    out = correlate(gym_scores, truth, bootstrap=0)
    assert out["spearman_rho"] > 0.85
    assert abs(out["spearman_top10"]) < 0.6
    # tau_ap sanity: identity = 1, reversal = -1
    a = np.arange(10, dtype=float)
    assert _tau_ap(a, a) == 1.0
    assert _tau_ap(-a, a) == -1.0


def test_local_corpus():
    import tempfile
    from gym import Gym, GymConfig

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a.txt").write_text("alpha doc about refunds")
        (root / "b.md").write_text("beta doc about shipping")
        cfg = GymConfig(task_name="my-docs", corpus_path=str(root), judge="mock",
                        judge_instruction="Prefer results that answer support questions.",
                        models=["m1", "m2"], output_dir=root / "out")
        g = Gym(cfg)
        corpus = g.load_corpus()
        assert set(corpus) == {"a.txt", "b.md"}
        assert "support questions" in g.judge.system

if __name__ == "__main__":
    print("Running MTEB Gym smoke tests...\n")
    test_json_extraction()
    test_query_gen_and_filter()
    test_search_retriever()
    test_env_knobs_rejected()
    ratings = test_judge_and_scoring()
    test_correlation(ratings)
    test_exact_permutation_p()
    test_parallel_judging_matches_sequential()
    test_parallel_query_gen_matches_sequential()
    test_identical_results_short_circuit()
    test_parse_failure_counter()
    test_winless_model_ranks_last()
    test_corpus_fingerprint_content_sensitive()
    test_matchup_resume_from_jsonl()
    test_repro_helpers()
    test_result_helpers()
    test_unified_gym_api()
    test_rank_agreement_api()
    test_rank_agreement_mteb_fallback()
    test_corpus_control_resolution()
    test_judge_instruction_resolution()
    test_top10_and_tau_ap()
    test_local_corpus()
    print("\nAll smoke tests passed.")
