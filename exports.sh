export PROJECT_ID=punim0478
export PROJ="/data/scratch/projects/$PROJECT_ID/fatitaheri/llm-serving"
export CODE="$PROJ/code"

export RESULTS="/data/gpfs/projects/$PROJECT_ID/$fatitaheri/annotations/llm-results"
mkdir -p "$RESULTS"

module load GCCcore/11.3.0
module load Python/3.10.4

python3 -m venv "$PROJ/client-env"
source "$PROJ/client-env/bin/activate"

python -m pip install -r "$CODE/requirements.txt"
python -m pip check
python "$CODE/pipeline.py" --help


export DATA_DIR="/data/gpfs/projects/punim0478/fatitaheri/annotations/data/harm_consensus_5_piiRev_filtered/harm_consensus_mapped_filtered.json"

python "$CODE/pipeline.py" \
  --data-dir "$DATA_DIR" \
  --output "$RESULTS/ingest-check.jsonl" \
  --audit-log "$RESULTS/ingest-check-audit.jsonl" \
  --model qwen3-30b-a3b \
  --skip-prefilter \
  --dry-run



sinteractive --partition=sapphire \
  --time=02:00:00 \
  --cpus-per-task=4 \
  --mem=32G


export PROJECT_ID=punim0478
export PROJ="/data/scratch/projects/$PROJECT_ID/fatitaheri/llm-serving"

module load GCCcore/11.3.0
module load Apptainer/1.1.8

export APPTAINER_CACHEDIR="$PROJ/apptainer-cache"
export APPTAINER_TMPDIR="$PROJ/build-tmp"
export UV_CACHE_DIR="$PROJ/build-tmp/uv-cache"

cd "$PROJ"

apptainer build \
  "$PROJ/vllm-qwen3.sif" \
  "$PROJ/vllm-qwen3-30b-a3b.def"