#!/usr/bin/env python3
"""
Correlate a gym leaderboard against ground truth.

    # against built-in MTEB retrieval scores for the default models:
    PYTHONPATH=. python3 scripts/validate.py --gym results/tournament/leaderboard.json --truth mteb

    # against your own {model: score} json (e.g. Arena ELO):
    PYTHONPATH=. python3 scripts/validate.py --gym results/.../leaderboard.json --truth arena.json
"""
import argparse
import json

from gym.validate import report

# Official MTEB NFCorpus nDCG@10 anchors (test split), verified against
# github.com/embeddings-benchmark/results on 2026-06-10. bm25 has no entry in
# the results repo; 32.5 is the commonly cited BEIR reference number.
# Use --fetch-truth to re-pull live instead of trusting these.
MTEB_NFCORPUS = {
    "bm25": 32.5,
    "sentence-transformers/all-MiniLM-L6-v2": 31.59,   # rev 8b3219a9
    "intfloat/multilingual-e5-small": 31.10,           # rev fd1525a9
    "intfloat/multilingual-e5-large-instruct": 36.34,  # rev baa7be48
    "BAAI/bge-large-en-v1.5": 38.06,                   # rev d4aa6901
    "nomic-ai/nomic-embed-text-v1.5": 34.67,
    "mixedbread-ai/mxbai-embed-large-v1": 38.67,       # rev 990580e2
}

_RESULTS_API = "https://api.github.com/repos/embeddings-benchmark/results/contents/results"
_RESULTS_RAW = "https://raw.githubusercontent.com/embeddings-benchmark/results/main/results"


def fetch_truth(models: list[str], task: str = "NFCorpus") -> dict[str, float]:
    """Pull official nDCG@10 for `models` from the embeddings-benchmark/results repo."""
    import urllib.request

    def _get_json(url):
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)

    out: dict[str, float] = {}
    for model in models:
        if "/" not in model:        # bm25 etc. have no results-repo entry
            continue
        slug = model.replace("/", "__")
        try:
            revisions = [e["name"] for e in _get_json(f"{_RESULTS_API}/{slug}")
                         if e["name"] != "model_meta.json"]
            # prefer a real revision hash over 'external' / 'no_revision_available'
            revisions.sort(key=lambda r: (r in ("external", "no_revision_available"), r))
            for rev in revisions:
                try:
                    data = _get_json(f"{_RESULTS_RAW}/{slug}/{rev}/{task}.json")
                    out[model] = round(data["scores"]["test"][0]["ndcg_at_10"] * 100, 2)
                    break
                except Exception:  # noqa: BLE001 - missing file for this revision
                    continue
        except Exception as e:  # noqa: BLE001
            print(f"  fetch-truth: no result for {model}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gym", required=True, help="path to gym leaderboard.json")
    ap.add_argument("--truth", default="mteb",
                    help="'mteb' for built-in anchors, or path to a {model: score} json")
    ap.add_argument("--fetch-truth", action="store_true",
                    help="fetch official scores live from embeddings-benchmark/results "
                         "instead of the built-in anchors (bm25 keeps its anchor)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    truth = MTEB_NFCORPUS if args.truth == "mteb" else json.loads(open(args.truth).read())
    if args.fetch_truth:
        fetched = fetch_truth(list(truth))
        print(f"fetched {len(fetched)} official scores from embeddings-benchmark/results")
        truth = {**truth, **fetched}
    res = report(args.gym, truth, bootstrap=args.bootstrap)

    if "error" in res:
        print("Cannot validate:", res["error"])
        print("shared models:", res.get("shared"))
        return

    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)
    print(f"models compared : {res['n_models']}")
    p_exact = res.get("spearman_p_exact")
    p_label = f"p_exact={p_exact:.4f}" if p_exact is not None else f"p={res['spearman_p']:.3f}"
    print(f"Spearman rho    : {res['spearman_rho']:.3f}  ({p_label})")
    lo, hi = res["spearman_ci95"]
    if lo is not None:
        print(f"  95% CI        : [{lo:.2f}, {hi:.2f}]")
    print(f"Kendall tau     : {res['kendall_tau']:.3f}  (p={res['kendall_p']:.3f})")
    print("-" * 60)
    print("gym ranking      :", " > ".join(m.split('/')[-1] for m in res["gym_ranking"]))
    print("ground-truth rank:", " > ".join(m.split('/')[-1] for m in res["truth_ranking"]))
    print("=" * 60)
    p_show = p_exact if p_exact is not None else res["spearman_p"]
    if p_show > 0.05:
        print("note: not significant at p<0.05 — with this few models a high rho can "
              "still be chance. More models tightens this much faster than more queries.")
    elif res["spearman_rho"] >= 0.85:
        print(f"rho={res['spearman_rho']:.2f} (significant): in line with the real-query "
              "baseline (0.86). Check the CI before calling the two settings equivalent.")


if __name__ == "__main__":
    main()
