#!/usr/bin/env bash
#SBATCH --job-name=annotation-sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=annotation-sft-%j.out

set -euo pipefail
umask 077

# Edit these settings.
ACCOUNT="YOUR_SPARTAN_ACCOUNT"
WORK_DIR="/scratch/punim0478/fatitaheri/llm-tuning/llamafactory"
DATA_FILE="/scratch/punim0478/fatitaheri/llm-tuning/llamafactory/data/train.json"
PYTHON="python3.11"

MODEL="meta-llama/Llama-3.2-3B-Instruct"
TEMPLATE="llama3"
MODEL_REVISION="main"       # Use a Hugging Face commit SHA for repeatable runs.
MODE="inspect"             # inspect, smoke, or train
CUTOFF_LEN=2048
SMOKE_ROWS=64

# Set INSTRUCTION_FIELD="" if you want the same instruction for every row.
INSTRUCTION_FIELD="instruction"
INPUT_FIELD="input"
OUTPUT_FIELD="output"
DEFAULT_INSTRUCTION="Annotate the text and return the result as JSON."

# Spartan allocation settings.
GPU_PARTITION="deeplearn"
GPU_QOS="gpgpudeeplearn"
INSPECT_TIME="00:30:00"
SMOKE_TIME="00:30:00"
TRAIN_TIME="04:00:00"
GPU_HOST_MEMORY="64G"

FACTORY_COMMIT="7af909522a951e3ad9f022ea6f88b6755257eaa5"

if [[ "${1:-}" != "_worker" ]]; then
    script="$(realpath -- "$0")"
    script_dir="$(dirname -- "$script")"
    [[ "$ACCOUNT" != "YOUR_SPARTAN_ACCOUNT" ]] || {
        echo "Set ACCOUNT in run-sft.sh." >&2
        exit 2
    }
    [[ "$WORK_DIR" == /* && "$DATA_FILE" == /* && -r "$DATA_FILE" ]] || {
        echo "Set an absolute WORK_DIR and a readable absolute DATA_FILE." >&2
        exit 2
    }

    job_args=(--account="$ACCOUNT")
    case "$MODE" in
        inspect)
            job_args+=(--mem=16G --time="$INSPECT_TIME")
            ;;
        smoke|train)
            job_time="$SMOKE_TIME"
            [[ "$MODE" != train ]] || job_time="$TRAIN_TIME"
            job_args+=(--partition="$GPU_PARTITION" --qos="$GPU_QOS"
                       --gres=gpu:1 --mem="$GPU_HOST_MEMORY" --time="$job_time")
            ;;
        *)
            echo "MODE must be inspect, smoke, or train." >&2
            exit 2
            ;;
    esac

    mkdir -p "$WORK_DIR"
    export PIP_CACHE_DIR="$WORK_DIR/pip-cache"
    venv="$WORK_DIR/venv-$FACTORY_COMMIT"
    if [[ ! -f "$venv/requirements-installed.txt" ]]; then
        "$PYTHON" -m venv "$venv"
        "$venv/bin/python" -m pip install \
            "llamafactory @ git+https://github.com/hiyouga/LLaMA-Factory.git@$FACTORY_COMMIT" \
            bitsandbytes
        "$venv/bin/python" -m pip check
        "$venv/bin/python" -m pip freeze > "$venv/requirements-installed.txt"
    fi

    exec sbatch "${job_args[@]}" "$script" _worker "$script_dir"
fi

: "${SLURM_JOB_ID:?Run this file with bash run-sft.sh to submit it}"
script_dir="$2"
venv="$WORK_DIR/venv-$FACTORY_COMMIT"
run_dir="$WORK_DIR/runs/$SLURM_JOB_ID"
export HF_HOME="${HF_HOME:-$WORK_DIR/huggingface}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$run_dir" "$HF_HOME"
cp -- "$0" "$run_dir/run-sft.sh"
cp -- "$script_dir/prepare_dataset.py" "$run_dir/prepare_dataset.py"
cp -- "$venv/requirements-installed.txt" "$run_dir/"

row_args=()
if [[ "$MODE" == smoke ]]; then
    row_args+=(--limit "$SMOKE_ROWS")
fi

"$venv/bin/python" "$run_dir/prepare_dataset.py" \
    --source "$DATA_FILE" --destination "$run_dir/data" \
    --instruction-field "$INSTRUCTION_FIELD" \
    --input-field "$INPUT_FIELD" --output-field "$OUTPUT_FIELD" \
    --default-instruction "$DEFAULT_INSTRUCTION" \
    --model "$MODEL" --template "$TEMPLATE" --revision "$MODEL_REVISION" \
    --cutoff "$CUTOFF_LEN" "${row_args[@]}"

if [[ "$MODE" == inspect ]]; then
    exit 0
fi

"$venv/bin/python" - "$run_dir/data/length_report.json" <<'PY'
import json
import sys
from pathlib import Path

import torch

report = json.loads(Path(sys.argv[1]).read_text())
if report["over_cutoff"]:
    raise SystemExit("Training stopped. Review length_report.json and adjust CUTOFF_LEN.")
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("This recipe requires a CUDA GPU with BF16 support.")
print(torch.cuda.get_device_name(0))
PY

max_steps=-1
[[ "$MODE" != smoke ]] || max_steps=5

exec "$venv/bin/llamafactory-cli" train \
    --stage sft --do_train true \
    --model_name_or_path "$MODEL" --model_revision "$MODEL_REVISION" \
    --template "$TEMPLATE" \
    --dataset_dir "$run_dir/data" --dataset annotation_train \
    --cutoff_len "$CUTOFF_LEN" --preprocessing_num_workers 4 \
    --finetuning_type lora --lora_rank 16 --lora_target all \
    --quantization_bit 4 --quantization_method bitsandbytes \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 --num_train_epochs 1 --max_steps "$max_steps" \
    --bf16 true --logging_steps 1 --save_strategy epoch --report_to none \
    --output_dir "$run_dir/adapter"