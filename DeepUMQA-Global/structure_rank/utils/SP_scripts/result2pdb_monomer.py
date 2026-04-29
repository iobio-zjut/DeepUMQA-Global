import os
import sys
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 参数解析 ---
def parse_args():
    parser = argparse.ArgumentParser(description="Extract template PDBs based on Foldseek m8 results.")
    parser.add_argument('--root', type=str, help='Monomer test root directory')
    parser.add_argument('--db', type=str, default=os.environ.get("DEEPUMQA_AFDB_DIR", ""), help='Path to template library')
    
    args, unknown = parser.parse_known_args()
    root_path = args.root if args.root else (sys.argv[1] if len(sys.argv) > 1 else None)
    if not root_path:
        print("❌ 错误: 请提供项目 root 路径")
        sys.exit(1)
    return os.path.abspath(root_path), args.db

# --- 核心提取函数 ---
def find_and_copy_pdb(target_id, db_dir, target_output_dir):
    """
    在数据库中查找 target_id.pdb 并拷贝到目标目录
    """
    # 常见的 AFDB 命名可能是 target_id 或 target_id.pdb
    source_path = os.path.join(db_dir, f"{target_id}.pdb")
    dest_path = os.path.join(target_output_dir, f"{target_id}.pdb")
    
    if os.path.exists(dest_path):
        return None # 已存在，跳过

    if os.path.exists(source_path):
        try:
            shutil.copy(source_path, dest_path)
            return f"✅ Copied: {target_id}"
        except Exception as e:
            return f"❌ Error copying {target_id}: {e}"
    else:
        # 如果数据库有子目录，可以考虑在这里增加 glob 搜索，但目前按标准路径处理
        return f"⚠️ Not found: {target_id}"

def process_m8_file(m8_path, db_dir, output_base):
    """
    解析 m8 文件，提取模板 ID 并执行拷贝
    """
    # 提取 Query ID 和 Group Name (从路径中获取)
    # 路径示例: root/search_result/result/{group}/{query_id}_results.m8
    query_id = os.path.basename(m8_path).replace("_results.m8", "")
    group_name = os.path.basename(os.path.dirname(m8_path))
    
    # 模板保存路径: root/templates_cat/{group}/{query_id}/
    target_output_dir = os.path.join(output_base, group_name, query_id)
    os.makedirs(target_output_dir, exist_ok=True)

    template_ids = set()
    with open(m8_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                # Foldseek 标准格式：[0]是Query, [1]是Target
                t_id = parts[1]
                template_ids.add(t_id)

    if not template_ids:
        return [f"Empty results for {query_id}"]

    results = []
    # 使用线程池加速文件拷贝
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(find_and_copy_pdb, tid, db_dir, target_output_dir) for tid in template_ids]
        for f in as_completed(futures):
            res = f.result()
            if res: results.append(res)
    
    return results

def main():
    ROOT, DB_DIR = parse_args()
    if not DB_DIR:
        print("❌ Missing AFDB template directory. Provide --db or set DEEPUMQA_AFDB_DIR.")
        return

    M8_DIR = os.path.join(ROOT, "search_result", "result")
    OUTPUT_BASE = os.path.join(ROOT, "templates_cat") # 保持与后续步骤路径一致

    if not os.path.exists(M8_DIR):
        print(f"❌ 找不到搜索结果目录: {M8_DIR}")
        return

    # 获取所有 m8 文件
    m8_files = []
    for root, _, files in os.walk(M8_DIR):
        for f in files:
            if f.endswith(".m8"):
                m8_files.append(os.path.join(root, f))

    print(f"🚀 发现 {len(m8_files)} 个搜索结果文件，开始提取结构...")

    for m8 in m8_files:
        print(f"📦 Processing: {os.path.basename(m8)}")
        logs = process_m8_file(m8, DB_DIR, OUTPUT_BASE)
        # 仅打印错误或关键信息，避免刷屏
        for log in logs:
            if "❌" in log or "⚠️" in log:
                print(f"  {log}")

    print(f"🎉 处理完成。结构存放在: {OUTPUT_BASE}")

if __name__ == "__main__":
    main()
