#!/usr/bin/env python3
"""
Judge quality report from persisted verdict files.

Usage:
    python scripts/judge_report.py [results_dir]

Reads every verdicts_*.json file in results_dir (default: results/) and prints
one row per model pair:

  Tie%      fraction of verdicts where score_a == 0.5 — the judge found no
            clear winner (genuine ties, identical result sets, or split orders)
  PosCons%  position consistency: among verdicts judged with both A-first and
            B-first orders, the fraction where flipping produced the same winner
            (100% = fully unbiased; lower values mean position-sensitive judge)
  a_first   fraction of decisive judgments that favored the first-shown system
            (0.50 = unbiased; values above ~0.55 indicate position bias)

Parse failure rate cannot be recovered per pair from verdict files (failures
are stored as "tie" in the raw field). The overall rate from leaderboard.json
is shown at the bottom when that file is present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _remap_flipped(w: str) -> str:
    """In the flipped order, 'A' means model_b won; map back to A/B space."""
    return {"A": "B", "B": "A", "tie": "tie"}.get(w, "tie")


def analyze(path: Path) -> dict | None:
    data = json.loads(path.read_text())
    if not data:
        return None
    n = len(data)

    # fraction where score_a == 0.5
    tie_n = sum(1 for v in data if v.get("score_a") == 0.5)

    # position consistency and a_first derived from raw per-order winners
    clean_two = 0 # two-order verdicts where both orders gave the same winner
    two_total = 0 # total two-order verdicts
    first = decisive = 0

    for v in data:
        raw = v.get("raw", [])
        for w in raw:
            if w in ("A", "B"):
                decisive += 1
                if w == "A":
                    first += 1
        if len(raw) == 2:
            two_total += 1
            w0 = raw[0]
            w1 = _remap_flipped(raw[1])
            if (w0 in ("A", "B") and w1 == w0) or (w0 == "tie" and w1 == "tie"):
                clean_two += 1

    v0 = data[0]
    return {
        "pair": f"{v0.get('model_a', '?')}  vs  {v0.get('model_b', '?')}",
        "n": n,
        "tie_pct": 100.0 * tie_n / n,
        "pos_cons_pct": 100.0 * clean_two / two_total if two_total else None,
        "a_first": first / decisive if decisive else None,
    }


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    files = sorted(results_dir.glob("verdicts_*.json"))
    if not files:
        print(f"No verdict files found in {results_dir}/")
        sys.exit(1)

    rows = [r for f in files if (r := analyze(f)) is not None]
    if not rows:
        print("All verdict files were empty.")
        sys.exit(1)

    pw = max(len(r["pair"]) for r in rows) + 2
    header = (f"{'Pair':<{pw}} {'N':>5}   {'Tie%':>6}  {'PosCons%':>9}  {'a_first':>7}")
    bar = "-" * len(header)
    print(f"\nJudge Quality Report  ({results_dir}/)")
    print("=" * len(header))
    print(header)
    print(bar)
    for r in rows:
        tie_s = f"{r['tie_pct']:>5.1f}%"
        pc_s = (f"{r['pos_cons_pct']:>8.1f}%" if r["pos_cons_pct"] is not None
                else "      n/a")
        af_s = f"{r['a_first']:.3f}" if r["a_first"] is not None else "    n/a"
        print(f"{r['pair']:<{pw}} {r['n']:>5}   {tie_s}  {pc_s}  {af_s}")
    print("=" * len(header))

    lb = results_dir/"leaderboard.json"
    if lb.exists():
        try:
            pfail = json.loads(lb.read_text()).get("parse_failure_rate")
            if pfail is not None:
                print(f"\nOverall parse failure rate: {pfail:.3f}  (from leaderboard.json)")
        except Exception:
            pass
    else:
        print(
            "\nNote: parse failure rate per pair is not available from verdict files "
            "(failures are recorded as ties). Run the full pipeline to get the overall "
            "rate in leaderboard.json."
        )

    print("""
Definitions:
  Tie%      score_a == 0.5: judge found no clear winner (true ties, identical
            result sets, or split orders all score 0.5)
  PosCons%  among two-order verdicts, fraction where flipping A/B gave the same
            winner (100% = fully position-consistent judge)
  a_first   fraction of decisive per-order judgments favoring the first-shown
            system (0.50 = unbiased; position bias inflates this above ~0.55)
""")


if __name__ == "__main__":
    main()
