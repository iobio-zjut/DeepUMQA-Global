import numpy as np
import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# ======================== 核心计算逻辑 ========================

def compute_structural_profile(query_dist_map, target_dist_maps, base_name, profile_dir):
    """计算距离分布直方图（Structural Profile）与熵"""
    all_maps = [query_dist_map] + target_dist_maps
    L = query_dist_map.shape[0]

    # 定义 36 个 bins: 2.0 到 20.0 Angstrom
    bins = np.linspace(2.0, 20.0, 37) 
    
    profile = np.zeros((L, L, 36), dtype=np.float32)
    entropy = np.zeros((L, L, 1), dtype=np.float32)
    mask = np.zeros((L, L), dtype=np.uint8)

    # 堆叠矩阵
    stacked_maps = np.stack(all_maps, axis=0)
    stacked_maps[stacked_maps <= 0] = np.nan

    i_indices, j_indices = np.triu_indices(L, k=1)

    for i, j in zip(i_indices, j_indices):
        distances = stacked_maps[:, i, j]
        valid_distances = distances[~np.isnan(distances)]
        
        if valid_distances.size == 0:
            continue

        hist, _ = np.histogram(valid_distances, bins=bins)
        sum_hist = hist.sum()
        
        if sum_hist > 0:
            prob = hist / sum_hist
            profile[i, j, :] = prob
            profile[j, i, :] = prob
            mask[i, j] = mask[j, i] = 1

            ent = -np.sum(prob * np.log(prob + 1e-8))
            entropy[i, j, 0] = entropy[j, i, 0] = ent

    out_file = os.path.join(profile_dir, f"{base_name}.npz")
    np.savez_compressed(
        out_file,
        profile=profile.astype(np.float16),
        entropy=entropy.astype(np.float16),
        mask=mask.astype(np.uint8)
    )
    return f"✅ Saved: {base_name}.npz"

# ======================== 任务分发逻辑 ========================

def process_single_query(folder, input_base, output_dirs):
    """处理单个 Query 文件夹下的所有 .npy 矩阵"""
    folder_path = os.path.join(input_base, folder)
    if not os.path.isdir(folder_path): return None

    # --- 改进的 Query 矩阵寻找逻辑 ---
    # 尝试多种可能的 Query 命名方式: {folder}_q.npy 或直接就是 {folder}.npy
    query_file = None
    possible_q_names = [f"{folder}_q.npy", f"{folder}.npy"]
    for p_name in possible_q_names:
        if os.path.exists(os.path.join(folder_path, p_name)):
            query_file = os.path.join(folder_path, p_name)
            break
    
    if not query_file:
        return f"⚠️ Missing Query Matrix in {folder} (Checked: {possible_q_names})"

    # --- 收集 Template 矩阵 ---
    # 排除掉 query 矩阵本身，其余全部作为 template
    target_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                    if f.endswith(".npy") and os.path.join(folder_path, f) != query_file]
    
    if not target_files:
        return f"⚠️ No Target Matrices found for {folder}"

    try:
        query_map = np.load(query_file)
        target_maps = []
        for f in target_files:
            try:
                m = np.load(f)
                if m.shape == query_map.shape:
                    target_maps.append(m)
            except: continue
            
        if not target_maps:
            return f"⚠️ No valid targets (shape mismatch or corrupted) for {folder}"

        return compute_structural_profile(query_map, target_maps, folder, output_dirs['profile'])
    except Exception as e:
        return f"❌ Error {folder}: {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='Monomer test root directory')
    args, _ = parser.parse_known_args()
    
    if not args.root and len(sys.argv) < 2:
        print("❌ 错误: 请提供 --root 参数")
        sys.exit(1)
        
    root = os.path.abspath(args.root if args.root else sys.argv[1])
    
    # 关键路径：必须与第四步输出对齐
    input_base = os.path.join(root, "CCdist") 
    output_root = os.path.join(root, "Profile")
    
    output_dirs = {
        'profile': os.path.join(output_root, "profile_npz")
    }
    for d in output_dirs.values(): os.makedirs(d, exist_ok=True)

    if not os.path.exists(input_base):
        print(f"❌ 错误: 输入路径不存在: {input_base}")
        return

    folders = sorted([f for f in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, f))])
    
    print(f"🧩 Found {len(folders)} folders in {input_base}")
    print(f"🚀 Processing Structural Profiles...")

    with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
        func = partial(process_single_query, input_base=input_base, output_dirs=output_dirs)
        results = list(executor.map(func, folders))
        
        # 统计结果
        success = 0
        for res in results:
            if res:
                print(res)
                if "✅" in res: success += 1
        
        print(f"\n✨ 任务结束! 成功生成: {success}/{len(folders)}")

if __name__ == "__main__":
    main()