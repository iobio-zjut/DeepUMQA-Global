#!/bin/bash

export LC_ALL=C

# --- 路径统一化处理 ---
# 接收第一个参数作为根目录
PROJECT_ROOT=$1

if [ -z "$PROJECT_ROOT" ]; then
    echo "❌ 错误: 请提供项目根目录路径"
    exit 1
fi

# 将路径转化为绝对路径，防止 cd 之后失效
PROJECT_ROOT=$(readlink -f "$PROJECT_ROOT")
FOLDSEEK_BIN="${DEEPUMQA_FOLDSEEK_BIN:-$(command -v foldseek || true)}"
TARGET_DB="${DEEPUMQA_SP_TEMPLATE_DB:-}"

# 检查 foldseek 命令是否存在
if [[ -z "$FOLDSEEK_BIN" ]]; then
    echo "❌ 错误: 未找到 foldseek 命令，请通过 PATH 或 DEEPUMQA_FOLDSEEK_BIN 提供"
    exit 1
fi
if [[ -z "$TARGET_DB" ]]; then
    echo "❌ 错误: 请设置 DEEPUMQA_SP_TEMPLATE_DB 指向 multimer Foldseek 数据库"
    exit 1
fi

# 统一化定义的路径
input_base_dir="$PROJECT_ROOT/pdb"
target_db="$TARGET_DB"
output_dir_base="$PROJECT_ROOT/search_result/result"
tmp_dir_base="$PROJECT_ROOT/search_result/tmp"

# 检查输入目录是否存在
if [ ! -d "$input_base_dir" ]; then
    echo "❌ 错误: 输入目录不存在: $input_base_dir"
    exit 1
fi

# 最大并发任务数
MAX_JOBS=32
job_count=0

# 创建必要目录
mkdir -p "$output_dir_base"
mkdir -p "$tmp_dir_base"

# 定义 foldseek 运行函数
run_foldseek() {
    local model_file="$1"
    local subdir_name="$2"
    
    model_name=$(basename "$model_file" .pdb)
    output_dir="$output_dir_base/$subdir_name"

    # 生成唯一临时目录防止并发冲突
    uuid=$(date +%s%N)_$RANDOM
    tmp_dir="$tmp_dir_base/$subdir_name/$uuid"
    result_file="$output_dir/${model_name}_results.m8"

    mkdir -p "$output_dir"
    mkdir -p "$tmp_dir"

    echo "🔍 [$(date +'%H:%M:%S')] 正在搜索: $model_name (源: $subdir_name)"

    # 执行 foldseek
    "$FOLDSEEK_BIN" easy-multimersearch "$model_file" "$target_db" "$result_file" "$tmp_dir" > /dev/null 2>&1
    
    # 检查是否成功生成结果
    if [ -f "$result_file" ]; then
        echo "✅ [$(date +'%H:%M:%S')] 完成: $model_name"
    else
        echo "⚠️ [$(date +'%H:%M:%S')] 失败: $model_name"
    fi

    # 清理临时目录
    [ -d "$tmp_dir" ] && rm -rf "$tmp_dir"
}

# --- 核心逻辑：遍历 pdb 目录下的 Target ---
echo "🚀 开始遍历目录: $input_base_dir"

# 使用 find 避免通配符在目录过大时崩溃
find "$input_base_dir" -mindepth 1 -maxdepth 1 -type d | while read -r subdir_path; do
    subdir=$(basename "$subdir_path")
    
    # 遍历 Target 目录下的 PDB
    for model_file in "$subdir_path"/*.pdb; do
        # 检查文件是否存在
        [ -f "$model_file" ] || continue

        model_name=$(basename "$model_file" .pdb)
        
        # 检查结果是否已存在 (断点续传逻辑)
        if [ -f "$output_dir_base/$subdir/${model_name}_results.m8" ]; then
            echo "⏩ 跳过 (结果已存在): $model_name"
            continue
        fi

        # 异步后台运行
        run_foldseek "$model_file" "$subdir" &

        ((job_count++))
        # 控制并发
        if ((job_count >= MAX_JOBS)); then
            wait
            job_count=0
        fi
    done
done

wait
echo "🏁 [$(date +'%Y-%m-%d %H:%M:%S')] 所有 foldseek 任务处理完毕。"




# #!/bin/bash

# export LC_ALL=C

# # 模块加载
# # module load anaconda
# # source activate foldseek-multimer-update

# # --- 路径统一化处理 ---
# PROJECT_ROOT=$1

# if [ -z "$PROJECT_ROOT" ]; then
#     echo "❌ 错误: 请提供项目根目录路径"
#     exit 1
# fi

# # 统一化定义的路径
# input_base_dir="$PROJECT_ROOT/pdb"
# output_dir_base="$PROJECT_ROOT/search_result/result"
# tmp_dir_base="$PROJECT_ROOT/search_result/tmp"

# # 最大并发任务数
# MAX_JOBS=32
# job_count=0

# # 创建必要目录
# mkdir -p "$output_dir_base"
# mkdir -p "$tmp_dir_base"

# # 定义 foldseek 运行函数
# run_foldseek() {
#     local model_file="$1"
#     local subdir_name="$2"
    
#     model_name=$(basename "$model_file" .pdb)
#     output_dir="$output_dir_base/$subdir_name"

#     # 生成唯一临时目录防止并发冲突
#     uuid=$(date +%s%N)_$RANDOM
#     tmp_dir="$tmp_dir_base/$subdir_name/$uuid"
#     result_file="$output_dir/${model_name}_results.m8"

#     mkdir -p "$output_dir"
#     mkdir -p "$tmp_dir"

#     echo "🔍 正在为 $model_file 搜索模板..."

#     # 执行 foldseek (确保路径变量加了引号以防空格导致 Input does not exist)
#     foldseek easy-multimersearch "$model_file" "$target_db" "$result_file" "$tmp_dir"

#     # 清理临时目录
#     [ -d "$tmp_dir" ] && rm -rf "$tmp_dir"
# }

# # --- 核心逻辑：遍历 pdb 目录下的 Target ---
# cd "$input_base_dir" || exit
# for subdir in *; do
#     # 确保是目录
#     if [ ! -d "$subdir" ]; then
#         continue
#     fi

#     # 遍历 Target 目录下的 PDB
#     for model_file in "$subdir"/*.pdb; do
#         # 检查文件是否存在
#         if [ ! -f "$model_file" ]; then
#             continue
#         fi

#         # 检查结果是否已存在
#         model_name=$(basename "$model_file" .pdb)
#         if [ -f "$output_dir_base/$subdir/${model_name}_results.m8" ]; then
#             echo "⏩ 跳过: $model_name"
#             continue
#         fi

#         # 传入完整路径（使用绝对路径避免 foldseek 找不到文件）
#         full_model_path="$(pwd)/$model_file"
        
#         run_foldseek "$full_model_path" "$subdir" &

#         ((job_count++))
#         if ((job_count >= MAX_JOBS)); then
#             wait
#             job_count=0
#         fi
#     done
# done

# wait
# echo "✅ 所有 foldseek 搜索任务完成。"
