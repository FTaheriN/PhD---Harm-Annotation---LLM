#!/usr/bin/env bash
set -euo pipefail

: "${LF_ROOT:?Set LF_ROOT to your Spartan project or scratch directory}"
: "${LF_REF:?Set LF_REF to a LLaMA-Factory commit SHA}"

mkdir -p "$LF_ROOT"
export PIP_CACHE_DIR="$LF_ROOT/pip-cache"

python3.11 -m venv "$LF_ROOT/venv"
python="$LF_ROOT/venv/bin/python"

"$python" -m pip install --upgrade pip
"$python" -m pip install \
    "llamafactory @ git+https://github.com/hiyouga/LLaMA-Factory.git@${LF_REF}" \
    bitsandbytes

"$python" -m pip check
"$LF_ROOT/venv/bin/llamafactory-cli" version
"$python" -m pip freeze > "$LF_ROOT/requirements-installed.txt"