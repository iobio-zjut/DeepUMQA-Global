#!/bin/bash
set -u

# =========================
# 0) Config from env
# =========================
base_dir="${PDB_BASE:?PDB_BASE not set}"
output_base_dir="${SEQ3DI_BASE:?SEQ3DI_BASE not set}"
log_base_dir="${LOG_BASE:?LOG_BASE not set}"
max_jobs="${MAX_JOBS:-24}"

processed_list="${log_base_dir}/codnas_seq3di_success.list"
failed_list="${log_base_dir}/codnas_seq3di_failed.list"
detail_log="${log_base_dir}/foldseek_detailed_errors.log"

lock_processed="${processed_list}.lock"
lock_failed="${failed_list}.lock"
lock_log="${detail_log}.lock"

mkdir -p "$output_base_dir"
mkdir -p "$log_base_dir"
touch "$processed_list" "$failed_list" "$detail_log"

# =========================
# Processing Function
# =========================
process_pdb() {
    local pdb_file="$1"
    local folder_name
    folder_name="$(realpath --relative-to="$base_dir" "$pdb_file" | cut -d'/' -f1)"

    local pdb_name
    pdb_name=$(basename "$pdb_file" .pdb)

    local output_dir="$output_base_dir/$folder_name"
    mkdir -p "$output_dir" 2>/dev/null || true

    local output_db="$output_dir/$pdb_name"
    local final_fasta="${output_db}_3di.fasta"

    # --- 1. Createdb ---
    if ! fs_log=$(foldseek createdb "$pdb_file" "$output_db" --threads 1 2>&1); then
        flock -x "$lock_failed" -c "echo '$pdb_file' >> '$failed_list'"
        flock -x "$lock_log" -c "echo '[ERROR-CREATEDB] $pdb_file: $fs_log' >> '$detail_log'"
        return 1
    fi

    # --- 2. Lndb ---
    rm -f "${output_db}_ss_h"
    if ! fs_log=$(foldseek lndb "${output_db}_h" "${output_db}_ss_h" 2>&1); then
        flock -x "$lock_failed" -c "echo '$pdb_file' >> '$failed_list'"
        flock -x "$lock_log" -c "echo '[ERROR-LNDB] $pdb_file: $fs_log' >> '$detail_log'"
        rm -f "${output_db}"* return 1
    fi

    # --- 3. Convert2fasta ---
    if ! fs_log=$(foldseek convert2fasta "${output_db}_ss" "$final_fasta" 2>&1); then
        flock -x "$lock_failed" -c "echo '$pdb_file' >> '$failed_list'"
        flock -x "$lock_log" -c "echo '[ERROR-CONVERT] $pdb_file: $fs_log' >> '$detail_log'"
        rm -f "${output_db}"*
        return 1
    fi

    # --- 4. 验证 & 清理数据库 ---
    if [[ ! -s "$final_fasta" ]]; then
        flock -x "$lock_failed" -c "echo '$pdb_file' >> '$failed_list'"
        rm -f "$final_fasta"
        return 1
    fi

    # 核心清理：只保留生成的 fasta，删除 foldseek 数据库二进制文件
    rm -f "${output_db}" "${output_db}.dbtype" "${output_db}_h" "${output_db}_h.dbtype" "${output_db}_h.index" "${output_db}.index" "${output_db}.lookup" "${output_db}_ss" "${output_db}_ss.dbtype" "${output_db}_ss.index" "${output_db}_ss_h" "${output_db}_ss_h.dbtype" "${output_db}_ss_h.index" 2>/dev/null || true

    flock -x "$lock_processed" -c "echo '$pdb_file' >> '$processed_list'"
    return 0
}

export -f process_pdb
export base_dir output_base_dir log_base_dir processed_list failed_list detail_log lock_processed lock_failed lock_log

# 使用 xargs 并行处理
find "$base_dir" -type f -name "*.pdb" -print0 | xargs -0 -r -P "$max_jobs" -I {} bash -c 'process_pdb "$1"' _ {}

echo "[INFO] Batch Finished."