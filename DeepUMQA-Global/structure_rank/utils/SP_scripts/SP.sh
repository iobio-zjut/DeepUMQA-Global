#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${1:-}"
PYTHON_BIN="${DEEPUMQA_PYTHON_BIN:-python}"

if [[ -z "${PROJECT_ROOT}" ]]; then
    echo "Error: please provide the SP working directory root." >&2
    exit 1
fi

PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

run_step() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

run_step "Step 1: Foldseek template search"
bash "${SCRIPT_DIR}/templateseek.sh" "${PROJECT_ROOT}"

run_step "Step 2: Downloading template structures"
"${PYTHON_BIN}" "${SCRIPT_DIR}/downpdb.py" --root "${PROJECT_ROOT}"

run_step "Step 3: Merging template/query structures"
"${PYTHON_BIN}" "${SCRIPT_DIR}/cat_pdb_cif_11.py" --root "${PROJECT_ROOT}"

run_step "Step 4: Structural alignment"
"${PYTHON_BIN}" "${SCRIPT_DIR}/align.py" --root "${PROJECT_ROOT}"

run_step "Step 5: Query distance matrices"
"${PYTHON_BIN}" "${SCRIPT_DIR}/CCdist_query.py" --root "${PROJECT_ROOT}"

run_step "Step 6: Template distance matrices"
"${PYTHON_BIN}" "${SCRIPT_DIR}/CCdist.py" --root "${PROJECT_ROOT}"

run_step "Step 7: Profile generation"
"${PYTHON_BIN}" "${SCRIPT_DIR}/profile.py" --root "${PROJECT_ROOT}"

run_step "SP multimer pipeline completed"
