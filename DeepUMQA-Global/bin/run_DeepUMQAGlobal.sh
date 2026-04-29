#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_PATH="${DEEPUMQA_IMAGE_PATH:-${SCRIPT_DIR}/deepumqa-g.sif}"
LOCAL_PROJECT_DIR="${DEEPUMQA_LOCAL_PROJECT:-/mydata/yeenjia/DeepUMQA-G/DeepUMQA-G_github}"
CONTAINER_PROJECT_MOUNT="/repo"
PYTHON_BIN_IN_CONTAINER="/opt/conda/envs/pytorch/bin/python"
FOLDSEEK_BIN_IN_CONTAINER="/opt/conda/envs/foldseek-multimer-update/bin/foldseek"
VORO_EXE_DIR_IN_CONTAINER="/opt/voronota/voronota-1.27.3834"

DEFAULT_SP_TEMPLATE_DB="/mydata/xielei/foldseek_db/PDB100/pdb"
DEFAULT_SP_MONOMER_TEMPLATE_DB="/mydata/xielei/xielei/MultiView_str-seq_profile/profile/PDB_AFDB_207187/PDB_AFDB_db"
DEFAULT_AFDB_DIR="/home/data/database/AFDB/PDB_AFDB_207187"

usage() {
  cat <<'EOF'
Usage:
  bash run_deepumqa.sh /path/to/case_dir [extra run_dual_inference.py args]

Required directory layout:
  case_dir/
    pdb/
      TARGET1/
        model1.pdb
        model2.pdb
    query/
      TARGET1/
        TARGET1.pdb

The script will create:
  case_dir/feature/
  case_dir/output/

Outputs:
  case_dir/output/global_score.csv
  case_dir/output/interface_score.csv

Default local project source:
  /mydata/yeenjia/DeepUMQA-G/DeepUMQA-G_github

Override with:
  DEEPUMQA_LOCAL_PROJECT=/path/to/your/project
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

CASE_DIR="$(realpath "$1")"
shift

PDB_ROOT="${CASE_DIR}/pdb"
QUERY_ROOT="${CASE_DIR}/query"
FEATURE_ROOT="${CASE_DIR}/feature"
OUTPUT_ROOT="${CASE_DIR}/output"

if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "Container image not found: ${IMAGE_PATH}" >&2
  exit 1
fi
if [[ ! -d "${LOCAL_PROJECT_DIR}" ]]; then
  echo "Missing local project directory: ${LOCAL_PROJECT_DIR}" >&2
  exit 1
fi
if [[ ! -d "${PDB_ROOT}" ]]; then
  echo "Missing pdb directory: ${PDB_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${QUERY_ROOT}" ]]; then
  echo "Missing query directory: ${QUERY_ROOT}" >&2
  exit 1
fi

mkdir -p "${FEATURE_ROOT}" "${OUTPUT_ROOT}"

for required_path in run_dual_inference.py checkpoints structure_rank; do
  if [[ ! -e "${LOCAL_PROJECT_DIR}/${required_path}" ]]; then
    echo "Local project directory is incomplete: missing ${LOCAL_PROJECT_DIR}/${required_path}" >&2
    exit 1
  fi
done

SP_TEMPLATE_DB="${DEEPUMQA_SP_TEMPLATE_DB:-${DEFAULT_SP_TEMPLATE_DB}}"
SP_MONOMER_TEMPLATE_DB="${DEEPUMQA_SP_MONOMER_TEMPLATE_DB:-${DEFAULT_SP_MONOMER_TEMPLATE_DB}}"
AFDB_DIR="${DEEPUMQA_AFDB_DIR:-${DEFAULT_AFDB_DIR}}"

for path in "${SP_TEMPLATE_DB}" "${SP_MONOMER_TEMPLATE_DB}" "${AFDB_DIR}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required database path not found: ${path}" >&2
    exit 1
  fi
done

SINGULARITY_ARGS=(
  --bind "${LOCAL_PROJECT_DIR}:${PROJECT_ROOT_IN_CONTAINER}"
  --bind "${CASE_DIR}:/work"
)

if command -v nvidia-smi >/dev/null 2>&1; then
  SINGULARITY_ARGS=(--nv "${SINGULARITY_ARGS[@]}")
fi

echo "Case directory: ${CASE_DIR}"
echo "Local project:  ${LOCAL_PROJECT_DIR}"
echo "PDB root:       /work/pdb"
echo "Query root:     /work/query"
echo "Feature root:   /work/feature"
echo "Output root:    /work/output"

cd /tmp

exec singularity exec "${SINGULARITY_ARGS[@]}" "${IMAGE_PATH}" \
  env \
    DEEPUMQA_PYTHON_BIN="${PYTHON_BIN_IN_CONTAINER}" \
    DEEPUMQA_FOLDSEEK_BIN="${FOLDSEEK_BIN_IN_CONTAINER}" \
    DEEPUMQA_MPNN_PYTHON="${PYTHON_BIN_IN_CONTAINER}" \
    DEEPUMQA_VORO_PYTHON="${PYTHON_BIN_IN_CONTAINER}" \
    DEEPUMQA_PYROSETTA_PYTHON="${PYTHON_BIN_IN_CONTAINER}" \
    DEEPUMQA_VORO_EXE_DIR="${VORO_EXE_DIR_IN_CONTAINER}" \
    "${PYTHON_BIN_IN_CONTAINER}" \
    "${PROJECT_ROOT_IN_CONTAINER}/run_dual_inference.py" \
    --pdb-root /work/pdb \
    --query-root /work/query \
    --feature-root /work/feature \
    --output-root /work/output \
    --ckpt-path "${PROJECT_ROOT_IN_CONTAINER}/checkpoints" \
    --python-bin "${PYTHON_BIN_IN_CONTAINER}" \
    --foldseek-bin "${FOLDSEEK_BIN_IN_CONTAINER}" \
    --mpnn-python "${PYTHON_BIN_IN_CONTAINER}" \
    --voro-python "${PYTHON_BIN_IN_CONTAINER}" \
    --pyrosetta-python "${PYTHON_BIN_IN_CONTAINER}" \
    --voro-exe-dir "${VORO_EXE_DIR_IN_CONTAINER}" \
    --sp-template-db "${SP_TEMPLATE_DB}" \
    --sp-monomer-template-db "${SP_MONOMER_TEMPLATE_DB}" \
    --afdb-dir "${AFDB_DIR}" \
    "$@"
