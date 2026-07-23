#!/usr/bin/env bash
# Run one ablation column (one judge/generator combo) over the NanoBEIR tasks.
#
# Works anywhere the gym repo is installed — a Slurm GPU node (see
# scripts/slurm/qwen27b_column.sbatch) or a laptop pointing at a remote
# endpoint through an SSH tunnel.
#
#   JUDGE_MODEL=Qwen/... TAG=judge-qwen27b ./scripts/run_column.sh
#
# Env vars:
#   JUDGE_MODEL  (required) exact model id as served by the endpoint
#   TAG          (required) short label for the column, e.g. judge-qwen27b
#   JUDGE_URL    OpenAI-compatible endpoint for the judge   [http://localhost:8000/v1]
#   GEN_MODEL    query-generator model id (family-disjoint from judge);
#                empty = generator falls back to the judge client
#   GEN_URL      endpoint for the generator                 [JUDGE_URL]
#   API_KEY_ENV  name of the env var holding the API key for both endpoints
#                (vLLM ignores it; Together API needs TOGETHER_API_KEY)
#   N_QUERIES    kept queries per task                      [50 — team nano convention]
#   WORKERS      concurrent judge calls                     [16]
#   MODELS       space-separated entrant roster override
#   OUTPUT_ROOT  results root                               [results/ablation]
#   TASKS        space-separated override of the task list
set -euo pipefail
cd "$(dirname "$0")/.."

JUDGE_MODEL="${JUDGE_MODEL:?set JUDGE_MODEL to the exact served model id}"
TAG="${TAG:?set TAG, e.g. judge-qwen27b}"
JUDGE_URL="${JUDGE_URL:-http://localhost:8000/v1}"
GEN_MODEL="${GEN_MODEL:-}"
GEN_URL="${GEN_URL:-$JUDGE_URL}"
API_KEY_ENV="${API_KEY_ENV:-}"
N_QUERIES="${N_QUERIES:-50}"
WORKERS="${WORKERS:-16}"
MODELS="${MODELS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/ablation}"

# The 13 NanoBEIR tasks (nano FinQA / AILACasedocs / BRIGHT-Biology don't
# exist yet — the team is creating those separately).
DEFAULT_TASKS="NanoNFCorpusRetrieval NanoFiQA2018Retrieval NanoSciFactRetrieval \
NanoArguAnaRetrieval NanoSCIDOCSRetrieval NanoTouche2020Retrieval \
NanoQuoraRetrieval NanoDBPediaRetrieval NanoHotpotQARetrieval \
NanoFEVERRetrieval NanoClimateFEVERRetrieval NanoNQRetrieval NanoMSMARCORetrieval"
read -r -a TASK_LIST <<< "${TASKS:-$DEFAULT_TASKS}"

# Fail fast if any task name doesn't exist in this mteb version.
PYTHONPATH=. python3 - "${TASK_LIST[@]}" <<'EOF'
import sys
import mteb
known = {t.metadata.name for t in mteb.get_tasks()}
missing = [name for name in sys.argv[1:] if name not in known]
if missing:
    nano = sorted(n for n in known if n.startswith("Nano"))
    sys.exit(f"unknown mteb tasks: {missing}\navailable Nano tasks: {nano}")
print(f"all {len(sys.argv) - 1} task names valid")
EOF

GEN_ARGS=()
if [[ -n "$GEN_MODEL" ]]; then
    GEN_ARGS+=(--gen-model "$GEN_MODEL" --gen-base-url "$GEN_URL")
fi
KEY_ARGS=()
if [[ -n "$API_KEY_ENV" ]]; then
    KEY_ARGS+=(--api-key-env "$API_KEY_ENV")
fi
MODEL_ARGS=()
if [[ -n "$MODELS" ]]; then
    read -r -a MODEL_LIST <<< "$MODELS"
    MODEL_ARGS=(--models "${MODEL_LIST[@]}")
fi

for task in "${TASK_LIST[@]}"; do
    out="$OUTPUT_ROOT/$TAG/$task"
    echo "=== $TAG / $task -> $out ==="
    PYTHONPATH=. python3 scripts/tournament.py \
        --judge qwen3 \
        --model "$JUDGE_MODEL" \
        --base-url "$JUDGE_URL" \
        --task "$task" \
        --n-queries "$N_QUERIES" \
        --workers "$WORKERS" \
        --output "$out" \
        "${MODEL_ARGS[@]}" "${GEN_ARGS[@]}" "${KEY_ARGS[@]}"
done

echo "column '$TAG' complete -> $OUTPUT_ROOT/$TAG/"
