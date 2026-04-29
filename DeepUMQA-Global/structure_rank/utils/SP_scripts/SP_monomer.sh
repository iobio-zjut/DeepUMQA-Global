#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${1:-}"
PYTHON_BIN="${DEEPUMQA_PYTHON_BIN:-python}"
AFDB_DIR="${DEEPUMQA_AFDB_DIR:-}"

if [[ -z "${ROOT}" ]]; then
    echo "Error: please provide the monomer SP working directory root." >&2
    exit 1
fi

ROOT="$(cd "${ROOT}" && pwd)"

if [[ -z "${AFDB_DIR}" ]]; then
    echo "Error: set DEEPUMQA_AFDB_DIR to the AFDB template directory." >&2
    exit 1
fi

log_step() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log_step "Step 1: Foldseek monomer search"
bash "${SCRIPT_DIR}/templateseek_monomer.sh" "${ROOT}"

log_step "Step 2: Extracting template PDB files"
"${PYTHON_BIN}" "${SCRIPT_DIR}/result2pdb_monomer.py" --root "${ROOT}" --db "${AFDB_DIR}"

log_step "Step 3: TM-align"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_tmalign_monomer.py" --root "${ROOT}"

log_step "Step 4: Distance matrices"
"${PYTHON_BIN}" "${SCRIPT_DIR}/CCdist_monomer.py" --root "${ROOT}"

log_step "Step 5: Profile synthesis"
"${PYTHON_BIN}" "${SCRIPT_DIR}/profile_monomer.py" --root "${ROOT}"

log_step "SP monomer pipeline completed"
