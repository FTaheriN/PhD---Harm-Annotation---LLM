#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash $0 DATASET [WALLTIME]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET="$(realpath -e -- "$1")"
WALLTIME="${2:-02:00:00}"

PROJ=/data/scratch/projects/punim0478/fatitaheri/llm-serving
IMAGE="$PROJ/vllm-0.17.1.sif"
MODEL_ID=meta-llama/Llama-3.3-70B-Instruct

[[ -r "$IMAGE" ]] || { echo "Cannot read image: $IMAGE" >&2; exit 1; }
[[ -r "$DATASET" ]] || { echo "Cannot read dataset: $DATASET" >&2; exit 1; }

export APPTAINERENV_HF_HUB_OFFLINE=1

sbatch --export=ALL \
    --job-name=llama33-70b-span \
    --partition=deeplearn \
    --account=punim0478 \
    --qos=gpgpudeeplearn \
    --constraint=dlg6 \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:H100:2 \
    --cpus-per-task=8 \
    --mem=192G \
    --time="$WALLTIME" \
    "$SCRIPT_DIR/test-span-model.sh" \
    "$MODEL_ID" 2 "$DATASET" "$IMAGE"


# bash submit-llama70b.sh /data/gpfs/projects/punim0478/fatitaheri/annotations/data/harm_consensus_5_piiRev_filtered/small_sample.json 00:30:00