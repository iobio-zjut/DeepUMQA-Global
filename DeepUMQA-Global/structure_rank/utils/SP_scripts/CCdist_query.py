import os
import sys
import numpy as np
import argparse
from Bio.PDB import PDBParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import pdist, squareform

# --- 参数解析 ---
def parse_args():
    parser = argparse.ArgumentParser(description="Calculate self-distance matrices for Query structures.")
    parser.add_argument('--root', type=str, help='Project root directory')
    args, unknown = parser.parse_known_args()
    
    root_path = args.root if args.root else (sys.argv[1] if len(sys.argv) > 1 else None)
    if not root_path:
        print("❌ 错误: 请提供项目 root 路径")
        sys.exit(1)
    return os.path.abspath(root_path)

# --- 核心提取逻辑 ---

def extract_all_residue_coords(pdb_file):
    """提取 PDB 中所有残基的坐标 (优先 CB, 次选 CA)，遵循物理顺序"""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("Q", pdb_file)
        if not structure: return np.array([])
        model = structure[0]
        coords = []
        for chain in model:
            for res in chain.get_residues():
                # 过滤异质原子
                if res.id[0] != ' ':
                    continue
                
                # 坐标提取：CB for all but Gly (CA)
                found_coord = None
                for atom_name in ["CB", "CA"]:
                    if atom_name in res:
                        found_coord = res[atom_name].get_coord()
                        break
                
                if found_coord is not None:
                    coords.append(found_coord)
                else:
                    coords.append([np.nan, np.nan, np.nan])
        return np.array(coords)
    except Exception as e:
        print(f"  [Error] Failed to parse {pdb_file}: {e}")
        return np.array([])

def calc_distance_matrix_fast(coords):
    """计算 $L \times L$ 距离矩阵，处理 NaN 坐标"""
    if len(coords) == 0:
        return np.array([])
    
    # 识别非空坐标
    mask = ~np.isnan(coords).any(axis=1)
    valid_coords = coords[mask]
    dist_matrix = np.full((len(coords), len(coords)), np.nan)

    if len(valid_coords) > 0:
        # 使用 scipy pdist 提升计算速度
        dists = squareform(pdist(valid_coords))
        valid_indices = np.where(mask)[0]
        # 填回原始维度的矩阵
        for i, idx in enumerate(valid_indices):
            dist_matrix[idx, valid_indices] = dists[i]
            
    return dist_matrix

def process_one_task(pdb_path, root_input, root_output):
    """处理单个 PDB 并保存，保持层级结构"""
    try:
        # 获取相对路径以保持目录结构: subdir/query_id/file.pdb
        rel_path = os.path.relpath(pdb_path, root_input)
        rel_dir = os.path.dirname(rel_path)
        base_name = os.path.splitext(os.path.basename(pdb_path))[0]
        
        output_dir = os.path.join(root_output, rel_dir)
        output_file = os.path.join(output_dir, f"{base_name}.npy")
        
        if os.path.exists(output_file):
            return f"EXISTS | {base_name}"

        coords = extract_all_residue_coords(pdb_path)
        if coords.size == 0:
            return f"EMPTY  | {base_name}"
            
        dist_matrix = calc_distance_matrix_fast(coords)
        
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_file, dist_matrix)
        
        return f"SUCCESS | {base_name} (Size: {dist_matrix.shape[0]})"
    except Exception as e:
        return f"FAILED  | {os.path.basename(pdb_path)}: {str(e)}"

# --- 主程序 ---

if __name__ == "__main__":
    ROOT = parse_args()
    root_query_cat = os.path.join(ROOT, "pdb")
    root_output_npy = os.path.join(ROOT, "query_CCdist")

    print(f"📂 项目根目录: {ROOT}")
    print("🔍 正在递归扫描结构文件...")
    
    pdb_files = []
    for dp, dn, filenames in os.walk(root_query_cat):
        for f in filenames:
            if f.endswith(".pdb"):
                pdb_files.append(os.path.join(dp, f))
    
    total = len(pdb_files)
    if total == 0:
        print("❌ 未在 query_cat 中找到 PDB 文件。")
        sys.exit(0)

    print(f"🚀 找到 {total} 个结构文件，启动 40 核并行计算...")

    with ProcessPoolExecutor(max_workers=min(40, os.cpu_count())) as executor:
        futures = [
            executor.submit(process_one_task, pdb_file, root_query_cat, root_output_npy)
            for pdb_file in pdb_files
        ]
        
        for i, future in enumerate(as_completed(futures), 1):
            if i % 50 == 0 or i == total: # 降低打印频率，每 50 个打印一次
                print(f"[{i}/{total}] {future.result()}")

    print("✨ 所有 Query 距离矩阵计算完成。")




# import os
# import sys
# import numpy as np
# from Bio.PDB import PDBParser
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from scipy.spatial.distance import pdist, squareform

# def extract_all_residue_coords(pdb_file):
#     """
#     提取 PDB 中所有残基的坐标 (CB, 若无则 CA)。
#     严格遵循 PDB 中的链和残基物理顺序。
#     """
#     parser = PDBParser(QUIET=True)
#     try:
#         structure = parser.get_structure("Q", pdb_file)
#         model = structure[0]
#         coords = []
#         for chain in model:
#             for res in chain.get_residues():
#                 # 过滤异质原子 (H_ 开头的残基)
#                 if res.id[0] != ' ':
#                     continue
                
#                 # 优先取 CB (甘氨酸无 CB，取 CA)
#                 found_coord = None
#                 for atom_name in ["CB", "CA"]:
#                     if atom_name in res:
#                         found_coord = res[atom_name].get_coord()
#                         break
                
#                 if found_coord is not None:
#                     coords.append(found_coord)
#                 else:
#                     coords.append([np.nan, np.nan, np.nan])
#         return np.array(coords)
#     except Exception as e:
#         print(f"  [Error] Failed to parse {pdb_file}: {e}")
#         return np.array([])

# def calc_distance_matrix_fast(coords):
#     """计算距离矩阵，处理含有 NaN 的情况"""
#     if len(coords) == 0:
#         return np.array([])
    
#     # 识别非空坐标的索引
#     mask = ~np.isnan(coords).any(axis=1)
#     valid_coords = coords[mask]
#     dist_matrix = np.full((len(coords), len(coords)), np.nan)

#     if len(valid_coords) > 0:
#         # 使用 pdist 快速计算成对距离
#         dists = squareform(pdist(valid_coords))
#         valid_indices = np.where(mask)[0]
#         # 将计算结果填回原始维度的矩阵中
#         for i, idx in enumerate(valid_indices):
#             dist_matrix[idx, valid_indices] = dists[i]
            
#     return dist_matrix

# def process_one_task(pdb_path, root_input, root_output):
#     """
#     处理单个 PDB 并保存为 npy。
#     输出路径保持与输入一致的层级结构。
#     """
#     try:
#         # 获取相对路径，例如: subdir/query_id/file.pdb
#         rel_path = os.path.relpath(pdb_path, root_input)
#         rel_dir = os.path.dirname(rel_path)
#         base_name = os.path.splitext(os.path.basename(pdb_path))[0]
        
#         output_dir = os.path.join(root_output, rel_dir)
#         output_file = os.path.join(output_dir, f"{base_name}.npy")
        
#         if os.path.exists(output_file):
#             return f"EXISTS | {base_name}"

#         coords = extract_all_residue_coords(pdb_path)
#         if coords.size == 0:
#             return f"EMPTY  | {base_name}"
            
#         dist_matrix = calc_distance_matrix_fast(coords)
        
#         os.makedirs(output_dir, exist_ok=True)
#         np.save(output_file, dist_matrix)
        
#         return f"SUCCESS | {base_name} (Size: {dist_matrix.shape[0]})"
#     except Exception as e:
#         return f"FAILED  | {os.path.basename(pdb_path)}: {str(e)}"

# if __name__ == "__main__":
#     # 路径对齐至 test 目录
#     root_query_cat = os.path.join(ROOT, "query_cat")
#     root_output_npy = os.path.join(ROOT, "query_CCdist")

#     # 深度收集所有 .pdb 文件 (适配 subdir/query_id/xxx.pdb)
#     print("🔍 正在扫描结构文件...")
#     pdb_files = glob_all_pdbs = [
#         os.path.join(dp, f) 
#         for dp, dn, filenames in os.walk(root_query_cat) 
#         for f in filenames if f.endswith(".pdb")
#     ]
    
#     total = len(pdb_files)
#     print(f"🚀 找到 {total} 个结构文件，启动多进程计算...")

#     # 使用多进程加速
#     with ProcessPoolExecutor(max_workers=40) as executor:
#         futures = [
#             executor.submit(process_one_task, pdb_file, root_query_cat, root_output_npy)
#             for pdb_file in pdb_files
#         ]
        
#         for i, future in enumerate(as_completed(futures), 1):
#             # 实时输出进度
#             print(f"[{i}/{total}] {future.result()}")

#     print("✨ 所有 Query 距离矩阵计算完成。")
