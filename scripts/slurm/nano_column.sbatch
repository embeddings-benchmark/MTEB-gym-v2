#!/bin/bash
# One nano ablation column on the Together/Laude cluster, as a thin client of
# the team's shared model servers (job names qwen36-serve / mmax27-serve) —
# do NOT start another 27B serve, per the warning in tejas/serve-qwen36-t.sbatch.
# Needs 1 GPU only, for encoding the embedding roster.
#
#   # run 1 (judge qwen3.6-27b, gen minimax-m2.7), all 13 nano tasks:
#   sbatch scripts/slurm/nano_column.sbatch
#
#   # smoke test:
#   sbatch --export=ALL,TAG=smoke,TASKS=NanoNFCorpusRetrieval,N_QUERIES=10 \
#       scripts/slurm/nano_column.sbatch
#
#   # run 2 (gen GLM-5.2 via Together API):
#   export TOGETHER_API_KEY=...
#   sbatch --export=ALL,TAG=judge-qwen27b_gen-glm52,GEN_MODEL=zai-org/GLM-5.2,\
#       GEN_URL=https://api.together.xyz/v1,API_KEY_ENV=TOGETHER_API_KEY \
#       scripts/slurm/nano_column.sbatch
#
#SBATCH --job-name=shriya-nanocol
#SBATCH --partition=slinky
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=/data/home/niklas/MTEB-gym-v2-embedding-repo/MTEB-gym-v2/logs/nanocol-%j.log

set -euo pipefail
REPO=/data/home/niklas/MTEB-gym-v2-embedding-repo/MTEB-gym-v2
cd "$REPO"
# PATH instead of `source activate`: activate hardcodes creation-time paths.
# Default env is tejas's roster env — the known-good cu128 torch stack on this
# cluster; the repo .venv has a cu130 torch the node driver can't run.
PYENV="${PYENV:-/data/home/niklas/tejas/envs/roster}"
export PATH="$PYENV/bin:$PATH"
mkdir -p logs

echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Iseconds)"

# Compute-node env fixes validated by the team's serve scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GYM_ENCODE_BATCH=1
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

# No cgroup GPU isolation on this cluster: the Slurm-assigned GPU may already
# be loaded by a colocated server job (this is what OOM'd the nanoval runs).
# Encode on the freest GPU with >=40GB free instead.
PICK=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
       | awk -F", *" '$2>40000{print $1" "$2}' | sort -k2 -rn | head -1 | awk '{print $1}')
[ -n "$PICK" ] || { echo "[FATAL] no GPU with >=40GB free on $(hostname)"; exit 1; }
export CUDA_VISIBLE_DEVICES=$PICK
echo "encoding on GPU $PICK"

# Judge: always the shared qwen3.6-27b serve
JNODE=$(squeue -h -n qwen36-serve -o %N | head -1)
[ -n "$JNODE" ] || { echo "[FATAL] qwen36-serve is not running"; exit 1; }
curl -sf "http://$JNODE:8000/v1/models" >/dev/null \
    || { echo "[FATAL] judge endpoint http://$JNODE:8000/v1 unhealthy"; exit 1; }

# Generator: shared minimax serve unless GEN_URL points elsewhere (e.g. Together API)
if [ -z "${GEN_URL:-}" ]; then
    GNODE=$(squeue -h -n mmax27-serve -o %N | head -1)
    [ -n "$GNODE" ] || { echo "[FATAL] mmax27-serve is not running"; exit 1; }
    GEN_URL="http://$GNODE:8000/v1"
fi
GEN_KEY="EMPTY"
if [ -n "${API_KEY_ENV:-}" ]; then GEN_KEY="${!API_KEY_ENV:-EMPTY}"; fi
curl -sf -o /dev/null -H "Authorization: Bearer $GEN_KEY" "$GEN_URL/models" 2>/dev/null \
    || echo "[WARN] generator endpoint $GEN_URL not reachable from compute node; proceeding"
echo "judge=http://$JNODE:8000/v1  gen=$GEN_URL"

# Team roster, verbatim from tejas/nano-validate.sbatch (UNION + NEW)
UNION="thenlper/gte-large mixedbread-ai/mxbai-embed-large-v1 intfloat/multilingual-e5-large-instruct BAAI/bge-base-en-v1.5 BAAI/bge-large-en-v1.5 BAAI/bge-small-en-v1.5 sentence-transformers/all-MiniLM-L6-v2 nomic-ai/nomic-embed-text-v1.5 bm25 sentence-transformers/all-mpnet-base-v2 intfloat/e5-base-v2 intfloat/multilingual-e5-small intfloat/e5-mistral-7b-instruct Salesforce/SFR-Embedding-2_R Alibaba-NLP/gte-Qwen2-7B-instruct thenlper/gte-small Snowflake/snowflake-arctic-embed-l-v2.0 Qwen/Qwen3-Embedding-8B Linq-AI-Research/Linq-Embed-Mistral"
NEW="Alibaba-NLP/gte-base-en-v1.5 Alibaba-NLP/gte-modernbert-base NovaSearch/stella_en_400M_v5 NovaSearch/stella_en_1.5B_v5 jinaai/jina-embeddings-v3 infly/inf-retriever-v1"

JUDGE_MODEL=qwen3.6-27b \
JUDGE_URL="http://$JNODE:8000/v1" \
GEN_MODEL="${GEN_MODEL:-minimax-m2.7}" \
GEN_URL="$GEN_URL" \
API_KEY_ENV="${API_KEY_ENV:-}" \
TAG="${TAG:-judge-qwen27b_gen-minimax}" \
N_QUERIES="${N_QUERIES:-50}" \
WORKERS=16 \
MODELS="${MODELS:-$UNION $NEW}" \
    bash scripts/run_column.sh

echo "done=$(date -Iseconds)"
