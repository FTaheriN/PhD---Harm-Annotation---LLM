#!/bin/bash -l
#SBATCH --job-name=annotation-sft
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=annotation-sft-%j.out

set -euo pipefail
umask 077

if (( $# < 4 || $# > 5 )); then
    echo "Usage: sbatch $0 DATA.json MODEL_OR_PATH TEMPLATE OUTPUT_ROOT [MAX_STEPS]" >&2
    exit 2
fi

: "${LF_ROOT:?Export LF_ROOT before submitting the job}"
data="$(realpath -e "$1")"
model="$2"
template="$3"
output_root="$4"
max_steps="${5:--1}"

if [[ ! -f "$data" || ! -r "$data" ]]; then
    echo "Dataset must be a readable JSON file." >&2
    exit 2
fi

export HF_HOME="${HF_HOME:-$LF_ROOT/huggingface}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$output_root"

work_dir="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/annotation-sft.XXXXXXXX")"
trap 'rm -rf -- "$work_dir"' EXIT

ln -s -- "$data" "$work_dir/train.json"
cat > "$work_dir/dataset_info.json" <<'JSON'
{
  "annotation_train": {
    "file_name": "train.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
JSON

nvidia-smi
"$LF_ROOT/venv/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable. Check the GPU allocation and PyTorch build.")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("This configuration requires a GPU with BF16 support.")
print(torch.cuda.get_device_name(0))
PY

"$LF_ROOT/venv/bin/llamafactory-cli" train \
    --stage sft \
    --do_train true \
    --model_name_or_path "$model" \
    --template "$template" \
    --dataset_dir "$work_dir" \
    --dataset annotation_train \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 16 \
    --quantization_bit 4 \
    --quantization_method bitsandbytes \
    --cutoff_len 2048 \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --max_steps "$max_steps" \
    --bf16 true \
    --logging_steps 1 \
    --save_strategy epoch \
    --report_to none \
    --output_dir "$output_root/$SLURM_JOB_ID"


# export LF_ROOT=/scratch/punim0478/fatitaheri/llm-tuning/llamafactory
# export LF_REF=7af909522a951e3ad9f022ea6f88b6755257eaa5
# bash setup-llamafactory.sh

# sbatch --account=YOUR_SPARTAN_ACCOUNT train-sft.sbatch \
#   /scratch/punim0478/fatitaheri/llm-tuning/llamafactory/data/smoke-train.json \
#   meta-llama/Llama-3.2-3B-Instruct llama3 \
#   /scratch/punim0478/fatitaheri/llm-tuning/llamafactory/runs 5