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

# Public MTEB English retrieval (nDCG@10, NFCorpus) anchors. Update from
# huggingface.co/spaces/mteb/leaderboard as the model set changes. These let you
# run a correlation immediately without scraping.
MTEB_NFCORPUS = {
    "bm25": 32.5,
    "sentence-transformers/all-MiniLM-L6-v2": 31.6,
    "intfloat/multilingual-e5-small": 31.9,
    "intfloat/multilingual-e5-large-instruct": 35.0,
    "BAAI/bge-large-en-v1.5": 34.5,
    "nomic-ai/nomic-embed-text-v1.5": 33.8,
    "mixedbread-ai/mxbai-embed-large-v1": 34.2,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gym", required=True, help="path to gym leaderboard.json")
    ap.add_argument("--truth", default="mteb",
                    help="'mteb' for built-in anchors, or path to a {model: score} json")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    truth = MTEB_NFCORPUS if args.truth == "mteb" else json.loads(open(args.truth).read())
    res = report(args.gym, truth, bootstrap=args.bootstrap)

    if "error" in res:
        print("Cannot validate:", res["error"])
        print("shared models:", res.get("shared"))
        return

    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)
    print(f"models compared : {res['n_models']}")
    print(f"Spearman rho    : {res['spearman_rho']:.3f}  (p={res['spearman_p']:.3f})")
    lo, hi = res["spearman_ci95"]
    if lo is not None:
        print(f"  95% CI        : [{lo:.2f}, {hi:.2f}]")
    print(f"Kendall tau     : {res['kendall_tau']:.3f}  (p={res['kendall_p']:.3f})")
    print("-" * 60)
    print("gym ranking      :", " > ".join(m.split('/')[-1] for m in res["gym_ranking"]))
    print("ground-truth rank:", " > ".join(m.split('/')[-1] for m in res["truth_ranking"]))
    print("=" * 60)
    if res["spearman_rho"] >= 0.85:
        print("rho >= 0.85: synthetic queries match ground truth as well as real ones.")
    elif res["spearman_p"] > 0.05:
        print("note: large p-value — with this few models the correlation is not yet significant.")


if __name__ == "__main__":
    main()
