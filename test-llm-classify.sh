#!/bin/bash -l
#SBATCH --job-name=qwen-classify
#SBATCH --account=punim0478
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg5
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=qwen-%j.out
#SBATCH --error=qwen-%j.err

set -euo pipefail

module purge
module load GCCcore/11.3.0
module load Apptainer/1.1.8
module load Python/3.10.4

PROJECT_ID="punim0478"
PROJ="/data/scratch/projects/$PROJECT_ID/fatitaheri/llm-serving"
CODE="$PROJ/code"
DATA_DIR="/data/gpfs/projects/punim0478/fatitaheri/annotations/data/harm_consensus_5_piiRev_filtered/harm_consensus_mapped_filtered.json"
RESULTS="/data/gpfs/projects/$PROJECT_ID/fatitaheri/llm-results"

source "$PROJ/client-env/bin/activate"
cd "$CODE"

mkdir -p "$PROJ/hf-cache" "$PROJ/logs" "$RESULTS"

# Check the main paths before starting the expensive server.
test -r "$PROJ/vllm-qwen3.sif"
test -r "$CODE/pipeline.py"
test -f "$DATA_DIR" && test -r "$DATA_DIR" || {
    echo "Dataset file missing or unreadable: $DATA_DIR" >&2
    exit 1
}

PORT=8000
SERVER_URL="http://127.0.0.1:$PORT"
SERVER_LOG="$PROJ/logs/vllm-${SLURM_JOB_ID}.log"

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

nvidia-smi

apptainer run --nv \
  --bind "$PROJ:$PROJ" \
  --env "HF_HOME=$PROJ/hf-cache" \
  --env HF_HUB_DISABLE_XET=1 \
  --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  "$PROJ/vllm-qwen3.sif" \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --served-model-name qwen3-30b-a3b-instruct-2507 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port "$PORT" \
  >"$SERVER_LOG" 2>&1 &

VLLM_PID=$!

echo "Waiting for vLLM. Log: $SERVER_LOG"

TIMEOUT=1800
SECONDS=0

until curl --fail --silent --show-error --max-time 5 \
    "$SERVER_URL/health" >/dev/null 2>&1; do

    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited. Check $SERVER_LOG"
        exit 1
    fi

    if (( SECONDS >= TIMEOUT )); then
        echo "vLLM startup timed out. Check $SERVER_LOG"
        exit 1
    fi

    sleep 5
done

echo "vLLM is ready."

curl --fail --silent --show-error "$SERVER_URL/v1/models"
echo

python "$CODE/pipeline.py" \
  --data-dir "$DATA_DIR" \
  --output "$RESULTS/test_classified-${SLURM_JOB_ID}.jsonl" \
  --audit-log "$RESULTS/test_prefilter_audit-${SLURM_JOB_ID}.jsonl" \
  --vllm-url "$SERVER_URL/v1" \
  --model qwen3-30b-a3b-instruct-2507 \
  --batch-size 4 \
  --context-window 0 \
  --thinking-budget 0 \
  --skip-prefilter \
  --log-file "$PROJ/logs/pipeline-${SLURM_JOB_ID}.log"
