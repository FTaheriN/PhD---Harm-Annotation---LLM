#!/bin/bash -l
#SBATCH --job-name=span-model
#SBATCH --account=punim0478
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

if [[ $# != 4 ]]; then
    echo "Usage: sbatch [GPU options] $0 MODEL_ID TENSOR_PARALLEL DATASET IMAGE" >&2
    exit 2
fi

MODEL_ID=$1
TP_SIZE=$2
DATASET=$3
IMAGE=$4
MODEL_NAME=${MODEL_ID##*/}

PROJECT_ID=punim0478
PROJ="/data/scratch/projects/$PROJECT_ID/fatitaheri/llm-serving/zero-shot"
RESULTS="/data/gpfs/projects/$PROJECT_ID/fatitaheri/llm-results"
HF_CACHE="$PROJ/hf-cache"
OUTPUT="$RESULTS/${MODEL_NAME}_${SLURM_JOB_ID}.jsonl"
SERVER_LOG="$PROJ/logs/${MODEL_NAME}_${SLURM_JOB_ID}_vllm.log"
PIPELINE_LOG="$PROJ/logs/${MODEL_NAME}_${SLURM_JOB_ID}_pipeline.log"

module purge
module load GCCcore/11.3.0 Apptainer/1.1.8 Python/3.10.4

PY="$PROJ/client-env/bin/python"
[[ -x "$PY" ]] || { echo "Python client missing: $PY" >&2; exit 1; }
[[ -r "$IMAGE" ]] || { echo "Image unreadable: $IMAGE" >&2; exit 1; }
[[ -r "$DATASET" ]] || { echo "Dataset unreadable: $DATASET" >&2; exit 1; }

mkdir -p "$HF_CACHE" "$RESULTS" "$PROJ/logs"

# Give separate jobs different ports if they share a node.
PORT=$((20000 + SLURM_JOB_ID % 30000))
SERVER_URL="http://127.0.0.1:$PORT"

MODEL_ARGS=()
if [[ "$MODEL_ID" == "mistralai/Ministral-3-14B-Instruct-2512" ]]; then
    MODEL_ARGS+=(--tokenizer-mode mistral --config-format mistral --load-format mistral)
fi

nvidia-smi

apptainer run --nv \
    --bind "$PROJ:$PROJ" \
    --env "HF_HOME=$HF_CACHE" \
    "$IMAGE" \
    --model "$MODEL_ID" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN:-8192}" \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.90 \
    --host 127.0.0.1 \
    --port "$PORT" \
    "${MODEL_ARGS[@]}" \
    >"$SERVER_LOG" 2>&1 &

VLLM_PID=$!
cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

SECONDS=0
until curl --fail --silent --max-time 5 "$SERVER_URL/health" >/dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null || (( SECONDS >= 1800 )); then
        echo "vLLM did not become ready. See $SERVER_LOG" >&2
        exit 1
    fi
    sleep 5
done

VLLM_MODERN_SPAN_API=1 "$PY" "$PROJ/code/pipeline.py" \
    --span-annotations \
    --task-id-field message_id \
    --data-dir "$DATASET" \
    --output "$OUTPUT" \
    --vllm-url "$SERVER_URL/v1" \
    --model "$MODEL_NAME" \
    --thinking-budget 0 \
    --log-file "$PIPELINE_LOG"

echo "Predictions: $OUTPUT"



#/data/gpfs/projects/punim0478/fatitaheri/annotations/data/harm_consensus_5_piiRev_filtered/small_sample.json
# SAMPLE=/data/gpfs/projects/punim0478/fatitaheri/annotations/data/harm_consensus_5_piiRev_filtered/small_sample.json
# IMAGE=/data/scratch/projects/punim0478/fatitaheri/llm-serving/vllm-0.17.1.sif

# sbatch --job-name=qwen35-9b-smoke \
#   --constraint=dlg4 --gres=gpu:a100-pcie-40gb:1 \
#   test-span-model.sh Qwen/Qwen3.5-9B 1 "$SAMPLE" "$IMAGE"


# module load GCCcore/11.3.0 Apptainer/1.1.8 Python/3.10.4

# PROJ=/data/scratch/projects/punim0478/fatitaheri/llm-serving
# IMAGE="$PROJ/vllm-0.17.1.sif"
# SAMPLE=/data/gpfs/projects/punim0478/fatitaheri/annotations/data/harm_consensus_5_piiRev_filtered/small_sample.json
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# APPTAINERENV_HF_HUB_OFFLINE=1 \
#   bash -l ./test-span-model.sh \
#   meta-llama/Llama-3.2-3B-Instruct 1 "$SAMPLE" "$IMAGE"

# "$PROJ/client-env/bin/python" "$PROJ/code/pipeline.py" \
#   --span-annotations --task-id-field message_id \
#   --data-dir "$SAMPLE" --dry-run

# apptainer exec --bind "$PROJ:$PROJ" \
#   --env "HF_HOME=$PROJ/hf-cache" \
#   --env HF_HUB_OFFLINE=1 \
#   "$IMAGE" python3 -c '
# import sys
# from vllm.transformers_utils.config import get_config
# config = get_config(sys.argv[1], trust_remote_code=False)
# print(config.model_type, config.architectures)
# ' "$MODEL"