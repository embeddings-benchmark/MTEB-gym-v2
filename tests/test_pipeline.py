"""
Smoke test: full pipeline on a tiny synthetic corpus with the MockClient.
No network, no GPU, no API key. Run: PYTHONPATH=. python3 tests/test_pipeline.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gym.baselines import BM25Retriever
from gym.clients import EnsembleClient, MockClient
from gym.config import GymConfig
from gym.judge import Judge, _parse_response
from gym.query_generator import Query, QueryGenerator, _extract_json
from gym.retrieval_harness import Retrieved
from gym.scoring import rate, format_leaderboard
from gym.validate import correlate
from gym.encoders import resolve_template, cache_key


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


def test_prefixes():
    assert resolve_template("intfloat/multilingual-e5-small").query.startswith("query:")
    assert resolve_template("intfloat/multilingual-e5-small").document.startswith("passage:")
    assert resolve_template("BAAI/bge-large-en-v1.5").query.startswith("Represent")
    assert resolve_template("sentence-transformers/all-MiniLM-L6-v2").query == "{text}"
    assert cache_key("intfloat/multilingual-e5-small") == "intfloat_multilingual-e5-small"
    # instruct-e5 must NOT fall through to the symmetric query:/passage: rule
    instruct = resolve_template("intfloat/multilingual-e5-large-instruct")
    assert instruct.query.startswith("Instruct:")
    assert instruct.document == "{text}"
    assert resolve_template("intfloat/e5-mistral-7b-instruct").query.startswith("Instruct:")
    print("  prefix registry + cache key ok")


def test_make_encoder_fallback():
    # unknown model -> mteb registry raises -> builtin Encoder, not a crash
    from gym.encoders import Encoder, make_encoder
    enc = make_encoder("not-a-real/model-xyz", task_name="NFCorpus")
    assert isinstance(enc, Encoder)
    assert make_encoder("any/model", task_name="NFCorpus", use_mteb=False).__class__ is Encoder
    print("  make_encoder fallback ok")


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


def test_bm25():
    corpus = make_corpus()
    queries = [Query("q0", "statins and heart disease evidence", ["D0"])]
    res = BM25Retriever(top_k=5).retrieve(corpus, queries)
    assert len(res) == 1 and len(res[0].doc_ids) == 5
    print("  bm25 retrieval ok")


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

    ratings = rate(verdicts, method="bradley_terry", bootstrap=200)
    assert len(ratings) == 3
    assert ratings[0].rating >= ratings[-1].rating
    assert ratings[0].ci >= 0
    print("  bradley-terry scoring ok")
    print(format_leaderboard(ratings))

    elo = rate(verdicts, method="elo", bootstrap=200)
    assert len(elo) == 3
    print("  online elo scoring ok")
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


def test_ensemble_vote():
    assert EnsembleClient.vote(["A", "A", "B"]) [0] == "A"
    assert EnsembleClient.vote(["A", "B"]) [0] == "tie"
    print("  ensemble vote ok")


def test_ensemble_judge_parse():
    # EnsembleClient.chat returns a JSON array of member responses; the judge
    # must majority-vote it instead of crashing/tieing (the drop-in-judge bug).
    members = ['{"winner": "A", "reasoning": "x"}',
               '{"winner": "A", "reasoning": "y"}',
               '{"winner": "B", "reasoning": "z"}']
    assert _parse_response(json.dumps(members)) == ("A", "x || y || z", True)
    # single (non-array) judge response still parses
    assert _parse_response('{"winner": "tie"}')[:2] == ("tie", "")
    # unparseable response is flagged, not silently scored
    assert _parse_response("no json here")[2] is False
    print("  ensemble judge parse ok")


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


if __name__ == "__main__":
    print("Running MTEB Gym smoke tests...\n")
    test_json_extraction()
    test_prefixes()
    test_make_encoder_fallback()
    test_query_gen_and_filter()
    test_bm25()
    test_ensemble_vote()
    test_ensemble_judge_parse()
    ratings = test_judge_and_scoring()
    test_correlation(ratings)
    test_exact_permutation_p()
    test_parallel_judging_matches_sequential()
    test_identical_results_short_circuit()
    test_parse_failure_counter()
    test_winless_model_ranks_last()
    print("\nAll smoke tests passed.")
