#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DEEPUMQA_PYTHON_BIN:-python}"
PDB_ROOT="${DEEPUMQA_PDB_ROOT:-${ROOT_DIR}/example/pdb}"
QUERY_ROOT="${DEEPUMQA_QUERY_ROOT:-${ROOT_DIR}/example/query}"
FEATURE_ROOT="${DEEPUMQA_FEATURE_ROOT:-${ROOT_DIR}/example/feature}"
OUTPUT_ROOT="${DEEPUMQA_OUTPUT_ROOT:-${ROOT_DIR}/example/output}"
CKPT_PATH="${DEEPUMQA_CKPT_PATH:-${ROOT_DIR}/checkpoints}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/run_dual_inference.py" \
  --pdb-root "${PDB_ROOT}" \
  --query-root "${QUERY_ROOT}" \
  --feature-root "${FEATURE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --ckpt-path "${CKPT_PATH}" \
  --python-bin "${PYTHON_BIN}" \
  "$@"
