#!/usr/bin/env python3
"""Aggregate one ablation column: validate every completed dataset leaderboard
against official full-dataset MTEB rankings and emit the paper-ready table.

    PYTHONPATH=. python3 scripts/aggregate_column.py \
        --root results/ablation/judge-qwen27b_gen-minimax \
        --out results/ablation/judge-qwen27b_gen-minimax/column.csv

Skips placeholder leaderboards (from query-pregen runs) by requiring the full
roster size. Ground truth is fetched live from embeddings-benchmark/results
for the *full-size* parent task of each nano dataset.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import fetch_truth, BM25_NDCG  # noqa: E402

from gym.validate import report  # noqa: E402

# nano task -> full-size parent task in the official results repo
PARENT_TASK = {
    "NanoNFCorpusRetrieval": "NFCorpus",
    "NanoFiQA2018Retrieval": "FiQA2018",
    "NanoSciFactRetrieval": "SciFact",
    "NanoArguAnaRetrieval": "ArguAna",
    "NanoSCIDOCSRetrieval": "SCIDOCS",
    "NanoTouche2020Retrieval": "Touche2020",
    "NanoQuoraRetrieval": "QuoraRetrieval",
    "NanoDBPediaRetrieval": "DBPedia",
    "NanoHotpotQARetrieval": "HotpotQA",
    "NanoFEVERRetrieval": "FEVER",
    "NanoClimateFeverRetrieval": "ClimateFEVER",
    "NanoNQRetrieval": "NQ",
    "NanoMSMARCORetrieval": "MSMARCO",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="column dir, e.g. results/ablation/<TAG>")
    ap.add_argument("--out", default=None, help="CSV output path (default <root>/column.csv)")
    ap.add_argument("--roster-size", type=int, default=25,
                    help="expected entrant count; smaller leaderboards are pregen placeholders")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for task, parent in PARENT_TASK.items():
        lb = root / task / "leaderboard.json"
        if not lb.exists():
            rows.append({"dataset": parent, "status": "missing"})
            continue
        d = json.loads(lb.read_text())
        models = [m["name"] for m in d.get("models", [])]
        if len(models) < args.roster_size:
            rows.append({"dataset": parent, "status": f"placeholder ({len(models)} models)"})
            continue
        truth = {"bm25": BM25_NDCG[parent]} if parent in BM25_NDCG and "bm25" in models else {}
        # cache fetched truth on disk: reruns (and rate-limit retries) are free
        tcache = root / task / f"truth_{parent}.json"
        fetched = json.loads(tcache.read_text()) if tcache.exists() else {}
        missing = [m for m in models if m not in fetched and "/" in m]
        if missing:
            fetched = {**fetched, **fetch_truth(missing, task=parent)}
            tcache.write_text(json.dumps(fetched, indent=1))
        truth = {**truth, **fetched}
        res = report(str(lb), truth, bootstrap=args.bootstrap)
        if "error" in res:
            rows.append({"dataset": parent, "status": f"error: {res['error']}"})
            continue
        lo, hi = res["spearman_ci95"]
        rows.append({
            "dataset": parent, "status": "ok",
            "n_models": res["n_models"],
            "spearman_rho": round(res["spearman_rho"], 3),
            "ci_low": None if lo is None else round(lo, 2),
            "ci_high": None if hi is None else round(hi, 2),
            "kendall_tau": round(res["kendall_tau"], 3),
            "a_first_rate": d.get("a_first_rate"),
            "parse_failure_rate": d.get("parse_failure_rate"),
        })
        print(f"{parent:15s} rho={res['spearman_rho']:.3f}  tau={res['kendall_tau']:.3f}  "
              f"n={res['n_models']}")

    out = Path(args.out) if args.out else root / "column.csv"
    fields = ["dataset", "status", "n_models", "spearman_rho", "ci_low", "ci_high",
              "kendall_tau", "a_first_rate", "parse_failure_rate"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    done = [r for r in rows if r.get("status") == "ok"]
    print(f"\n{len(done)}/{len(PARENT_TASK)} datasets validated -> {out}")
    if done:
        vals = [r["spearman_rho"] for r in done]
        print(f"mean spearman over completed: {sum(vals)/len(vals):.3f}")


if __name__ == "__main__":
    main()
