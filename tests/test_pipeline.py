"""
Smoke test: full pipeline on a tiny synthetic corpus with the MockClient.
No network, no GPU, no API key. Run: PYTHONPATH=. python3 tests/test_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gym.baselines import BM25Retriever
from gym.clients import EnsembleClient, MockClient
from gym.config import GymConfig
from gym.judge import Judge
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


def test_ensemble_vote():
    assert EnsembleClient.vote(["A", "A", "B"]) [0] == "A"
    assert EnsembleClient.vote(["A", "B"]) [0] == "tie"
    print("  ensemble vote ok")


if __name__ == "__main__":
    print("Running MTEB Gym smoke tests...\n")
    test_json_extraction()
    test_prefixes()
    test_query_gen_and_filter()
    test_bm25()
    test_ensemble_vote()
    ratings = test_judge_and_scoring()
    test_correlation(ratings)
    print("\nAll smoke tests passed.")
