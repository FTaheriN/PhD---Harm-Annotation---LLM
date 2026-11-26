# Annotation tuning on Spartan

## Before the first run

Keep `run-sft.sh` and `prepare_dataset.py` together. Load a Spartan Python
module that provides Python 3.11, or set `PYTHON` in `run-sft.sh` to the
absolute path of a compatible interpreter. The launcher creates a dedicated
environment in `WORK_DIR` and records the installed packages. It needs access
to GitHub and the Python package index on the first run.

Set `ACCOUNT`, `WORK_DIR`, `DATA_FILE`, and the dataset field names at the top
of `run-sft.sh`. Use a private project or scratch directory because converted
training records are written there. The source dataset is read only.

The source may be a JSON array or JSONL. Each record must contain an
instruction, an input string, and an output string, object, or array.
Set `INSTRUCTION_FIELD=""` to use `DEFAULT_INSTRUCTION` for every row.
Keep evaluation records outside the training file.

For gated checkpoints, obtain access and authenticate on Spartan before
submitting, or set `MODEL` to a local checkpoint directory readable by the
job. For repeatable runs, set `MODEL_REVISION` to the Hugging Face checkpoint
commit SHA or use a fixed local snapshot.

## Inspect token lengths

Start with `MODE="inspect"` and run:

```bash
bash run-sft.sh
```

Open `WORK_DIR/runs/<job-id>/data/length_report.json`. The report gives the
median, p95, p99, maximum, and how many rows exceed `CUTOFF_LEN`. Check the
report again for each model you plan to train because counts depend on the
tokenizer and template.

In LLaMA-Factory, `cutoff_len` limits the combined prompt and answer during
training. The response-token statistics describe your training targets and
can help size a separate inference `max_new_tokens` setting.

If the maximum fits GPU memory, set `CUTOFF_LEN` at or above that maximum to
preserve every example. If a few rows are much longer, review the reported
rows before deciding whether to shorten or exclude them. The launcher stops
training when any selected row exceeds the configured cutoff.

## Smoke train, then run the full dataset

Set `MODE="smoke"` and rerun the same command. This uses the first
`SMOKE_ROWS` records and runs five optimizer steps. Check the Slurm log for
training loss and `WORK_DIR/runs/<job-id>/adapter/` for adapter files.

Set `MODE="train"` and rerun the same command for one epoch over the entire
file. Adjust `TRAIN_TIME` for the workload. The run directory contains the
launcher, converter, installed package versions, converted training data,
length report, and adapter.

Start with Llama 3.2 3B. The GPU recipe uses one BF16-capable NVIDIA GPU and
bitsandbytes 4-bit QLoRA. Larger models and newer architectures need their own
smoke run to verify memory and quantization support. The 70B model may need a
high-memory GPU or a separate sharded training recipe. Training loss and an
adapter file confirm the job ran. Evaluate the resulting adapter on held-out
records to measure annotation quality.