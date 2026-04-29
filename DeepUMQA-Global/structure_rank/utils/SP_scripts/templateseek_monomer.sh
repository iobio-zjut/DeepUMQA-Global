#!/bin/bash

# 设置 locale，避免字符编码警告
export LC_ALL=C

# --- 核心：路径获取逻辑 ---
# 1. 如果有参数传入 ($1)，则使用参数
# 2. 如果没有参数，则自动推导当前脚本所在目录的父目录
if [ ! -z "$1" ]; then
    ROOT="$1"
    echo "📥 Received ROOT from parent script: $ROOT"
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    ROOT="$( dirname "$SCRIPT_DIR" )"
    echo "🏠 No parameter received. Auto-detected ROOT: $ROOT"
fi

# 将 ROOT 转换为绝对路径，防止相对路径在后续执行中失效
ROOT="$(cd "$ROOT"; pwd)"
FOLDSEEK_BIN="${DEEPUMQA_FOLDSEEK_BIN:-$(command -v foldseek || true)}"
TARGET_DB="${DEEPUMQA_SP_MONOMER_TEMPLATE_DB:-}"

# --- 目录定义 (基于 ROOT) ---
INPUT_BASE_DIR="$ROOT/pdb"
OUTPUT_DIR_BASE="$ROOT/search_result/result"
TMP_DIR_BASE="$ROOT/search_result/tmp"

# 模板保留配置
TOP_N=200
MAX_JOBS=32 

echo "📂 Project Root: $ROOT"
echo "🧬 Searching structures in: $INPUT_BASE_DIR"

# 检查输入目录
if [ ! -d "$INPUT_BASE_DIR" ]; then
    echo "❌ 错误: 输入目录 $INPUT_BASE_DIR 不存在"
    exit 1
fi
if [[ -z "$FOLDSEEK_BIN" ]]; then
    echo "❌ 错误: 未找到 foldseek 命令，请通过 PATH 或 DEEPUMQA_FOLDSEEK_BIN 提供"
    exit 1
fi
if [[ -z "$TARGET_DB" ]]; then
    echo "❌ 错误: 请设置 DEEPUMQA_SP_MONOMER_TEMPLATE_DB 指向 monomer Foldseek 数据库"
    exit 1
fi

# 创建基础目录
mkdir -p "$OUTPUT_DIR_BASE"
mkdir -p "$TMP_DIR_BASE"

job_count=0

# 遍历 group 目录
for target_group_dir in "$INPUT_BASE_DIR"/*; do
    [ -d "$target_group_dir" ] || continue
    group_name=$(basename "$target_group_dir")
    
    for model_file in "$target_group_dir"/*.pdb; do
        [ -f "$model_file" ] || continue
        
        model_name=$(basename "$model_file" .pdb)
        output_dir="$OUTPUT_DIR_BASE/$group_name"
        result_file="$output_dir/${model_name}_results.m8"
        tmp_dir="$TMP_DIR_BASE/$group_name/$model_name"

        if [ -f "$result_file" ]; then
            echo "⏩ Skip: $model_name (Result exists)"
            continue
        fi

        mkdir -p "$output_dir"
        mkdir -p "$tmp_dir"

        echo "🚀 Searching: $model_name in group $group_name"
        
        # --- 核心搜索任务 ---
        (
            # 注意：此处必须使用 $TARGET_DB (大写)，且移除静默输出以便调试，或仅重定向标准输出
            "$FOLDSEEK_BIN" easy-search "$model_file" "$TARGET_DB" "$result_file.raw" "$tmp_dir" \
                --format-mode 0 --threads 4 > "$tmp_dir/foldseek.log" 2>&1
            
            if [ -f "$result_file.raw" ]; then
                # 直接保留排序后的前 TOP_N 条命中。
                # 旧版按第 3 列做硬裁会丢掉 identity 偏低但结构上仍然合理的模板。
                awk 'NF >= 2 {print $0}' "$result_file.raw" | head -n "$TOP_N" > "$result_file"
                rm "$result_file.raw"
            fi
            
            # 清理
            rm -rf "$tmp_dir"
        ) &

        ((job_count++))
        if ((job_count >= MAX_JOBS)); then
            wait
            job_count=0
        fi
    done
done

wait
echo "✨ Foldseek Monomer Search 完成。"
