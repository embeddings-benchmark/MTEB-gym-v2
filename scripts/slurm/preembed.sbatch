#!/bin/bash
# Pre-encode every roster model's corpus embeddings for one task, one model per
# python process (tejas's preembed pattern) — process exit guarantees GPU
# release, defeating in-process accumulation OOMs. The tournament then hits the
# cache and runs light.
#
#   sbatch --export=ALL,PTASK=NanoTouche2020Retrieval scripts/slurm/preembed.sbatch
#
#SBATCH --job-name=shriya-preembed
#SBATCH --partition=slinky
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=/data/home/niklas/MTEB-gym-v2-embedding-repo/MTEB-gym-v2/logs/preembed-%j.log
set -u
PTASK="${PTASK:?export PTASK, e.g. NanoTouche2020Retrieval}"
REPO=/data/home/niklas/MTEB-gym-v2-embedding-repo/MTEB-gym-v2
OUT="$REPO/results/ablation/${PTAG:-judge-qwen27b_gen-minimax}/$PTASK"
cd "$REPO"
export PATH=/data/home/niklas/tejas/envs/roster/bin:$PATH
export PYTHONPATH=$REPO
export GYM_ENCODE_BATCH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GYM_MAX_DOC_CHARS=20000

MODELS="thenlper/gte-large mixedbread-ai/mxbai-embed-large-v1 intfloat/multilingual-e5-large-instruct BAAI/bge-base-en-v1.5 BAAI/bge-large-en-v1.5 BAAI/bge-small-en-v1.5 sentence-transformers/all-MiniLM-L6-v2 nomic-ai/nomic-embed-text-v1.5 sentence-transformers/all-mpnet-base-v2 intfloat/e5-base-v2 intfloat/multilingual-e5-small intfloat/e5-mistral-7b-instruct Salesforce/SFR-Embedding-2_R Alibaba-NLP/gte-Qwen2-7B-instruct thenlper/gte-small Snowflake/snowflake-arctic-embed-l-v2.0 Qwen/Qwen3-Embedding-8B Linq-AI-Research/Linq-Embed-Mistral Alibaba-NLP/gte-base-en-v1.5 Alibaba-NLP/gte-modernbert-base NovaSearch/stella_en_400M_v5 NovaSearch/stella_en_1.5B_v5 jinaai/jina-embeddings-v3 infly/inf-retriever-v1"

for m in $MODELS; do
    # per-model GPU re-pick (tejas): one pick at job start leaves hours of
    # exposure for co-tenants to fill the GPU
    line=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1)
    idx=${line%%,*}; free=$(echo ${line##*,} | tr -d " ")
    [ "${free:-0}" -ge 40000 ] && export CUDA_VISIBLE_DEVICES=$idx
    echo "=== preembed $m @ $(date -Iseconds) [gpu ${CUDA_VISIBLE_DEVICES:-?} ${free}MiB]"
    python scripts/encode_one.py "$m" "$PTASK" "$OUT" || echo "[WARN] preembed failed for $m"
done
echo "=== preembed $PTASK done $(date -Iseconds)"
