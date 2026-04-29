import os
import sys
import gzip
import shutil
import requests
import argparse
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 参数与路径解析 ---
def parse_args():
    parser = argparse.ArgumentParser(description="Download PDB/CIF templates based on Foldseek results.")
    # 同时支持 --root 方式和直接传参方式
    parser.add_argument('--root', type=str, help='Project root directory')
    args, unknown = parser.parse_known_args()
    
    # 如果没有 --root，尝试取第一个位置参数
    root_path = args.root if args.root else (sys.argv[1] if len(sys.argv) > 1 else None)
    
    if not root_path:
        print("❌ 错误: 请提供项目根目录路径 (使用 --root 或直接跟在脚本后)")
        sys.exit(1)
    
    # 转为绝对路径
    return os.path.abspath(root_path)

project_root = parse_args()
report_root = os.path.join(project_root, "search_result", "result")
output_root = os.path.join(project_root, "templates")

os.makedirs(output_root, exist_ok=True)

# RCSB 镜像地址
RCSB_PDB_URL = "https://files.rcsb.org/download/{}.pdb{}.gz"
RCSB_CIF_URL = "https://files.rcsb.org/download/{}-assembly{}.cif.gz"

def cif_to_pdb(cif_path, pdb_path):
    """将 cif 转换为 pdb 并删除原文件"""
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("structure", cif_path)
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_path)
        if os.path.exists(pdb_path):
            os.remove(cif_path) # 转换成功后删除 cif
            return True
    except Exception as e:
        print(f"❌ CIF 转换失败 {cif_path}: {e}")
        return False

def download_and_process(pdb_id: str, assembly_num: str, save_dir: str):
    pdb_id_upper = pdb_id.upper()
    # 目标文件名统一为 .pdb{n}
    final_pdb_name = f"{pdb_id_upper}.pdb{assembly_num}"
    final_pdb_path = os.path.join(save_dir, final_pdb_name)

    if os.path.exists(final_pdb_path):
        return

    # 1. 尝试直接下载 PDB
    pdb_url = RCSB_PDB_URL.format(pdb_id_upper, assembly_num)
    gz_path = final_pdb_path + ".gz"
    
    try:
        r = requests.get(pdb_url, timeout=20, stream=True)
        if r.status_code == 200:
            with open(gz_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            with gzip.open(gz_path, "rb") as f_in, open(final_pdb_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(gz_path)
            print(f"✅ Downloaded PDB: {final_pdb_name}")
            return
    except:
        if os.path.exists(gz_path): os.remove(gz_path)
        pass

    # 2. 如果 PDB 不存在，下载 CIF 并转换
    cif_url = RCSB_CIF_URL.format(pdb_id_upper, assembly_num)
    tmp_cif_path = os.path.join(save_dir, f"{pdb_id_upper}_{assembly_num}_tmp.cif")
    cif_gz_path = tmp_cif_path + ".gz"

    try:
        r = requests.get(cif_url, timeout=20, stream=True)
        if r.status_code == 200:
            with open(cif_gz_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            with gzip.open(cif_gz_path, "rb") as f_in, open(tmp_cif_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(cif_gz_path)
            
            # 转换
            if cif_to_pdb(tmp_cif_path, final_pdb_path):
                print(f"🔄 Downloaded CIF & Converted to: {final_pdb_name}")
            if os.path.exists(tmp_cif_path): os.remove(tmp_cif_path)
        else:
            print(f"❌ Not Found: {pdb_id_upper} Assembly {assembly_num}")
    except Exception as e:
        if os.path.exists(cif_gz_path): os.remove(cif_gz_path)
        print(f"⚠️ Error: {pdb_id_upper} -> {e}")

# --- 任务收集逻辑 ---
def collect_tasks():
    tasks = set()
    if not os.path.exists(report_root):
        print(f"⚠️ 警告: 报告目录不存在 {report_root}")
        return tasks

    for query_name in os.listdir(report_root):
        query_dir = os.path.join(report_root, query_name)
        if not os.path.isdir(query_dir): continue
        for file_name in os.listdir(query_dir):
            if not file_name.endswith(".m8"): continue
            
            m8_path = os.path.join(query_dir, file_name)
            try:
                with open(m8_path, "r") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        # 识别格式: 1abc-assembly1.cif 或 1abc.pdb1
                        if len(parts) < 2: continue
                        target_id = parts[1]
                        
                        if "-assembly" in target_id:
                            p_id = target_id.split("-")[0]
                            a_num = target_id.split("-")[1].replace("assembly", "").split(".")[0]
                        elif ".pdb" in target_id:
                            p_id = target_id.split(".")[0]
                            a_num = target_id.split(".")[1].replace("pdb", "")
                        else:
                            continue

                        if not os.path.exists(os.path.join(output_root, f"{p_id.upper()}.pdb{a_num}")):
                            tasks.add((p_id, a_num, output_root))
            except Exception as e:
                print(f"读取 {file_name} 出错: {e}")
    return tasks

# --- 主执行流程 ---
if __name__ == "__main__":
    print(f"📂 项目根目录: {project_root}")
    print(f"🔍 正在扫描 Foldseek 结果...")
    
    download_tasks = collect_tasks()

    if download_tasks:
        print(f"🚀 开始处理 {len(download_tasks)} 个模板下载任务...")
        # 增加超时控制，防止进程卡死
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(download_and_process, p, a, o) for p, a, o in download_tasks]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"任务执行异常: {e}")
        print("\n✅ 所有模板下载与转换尝试完成。")
    else:
        print("✨ 未发现需要下载的新模板。")



# import os
# import sys
# import gzip
# import shutil
# import requests
# from Bio.PDB.MMCIFParser import MMCIFParser
# from Bio.PDB.PDBIO import PDBIO
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # --- 路径设置 ---
# if len(sys.argv) < 2:
#     print("❌ 错误: 请提供项目根目录路径")
#     sys.exit(1)

# project_root = sys.argv[1]
# report_root = os.path.join(project_root, "search_result", "result")
# output_root = os.path.join(project_root, "templates")

# os.makedirs(output_root, exist_ok=True)

# RCSB_PDB_URL = "https://files.rcsb.org/download/{}.pdb{}.gz"
# RCSB_CIF_URL = "https://files.rcsb.org/download/{}-assembly{}.cif.gz"

# def cif_to_pdb(cif_path, pdb_path):
#     """将 cif 转换为 pdb 并删除原文件"""
#     try:
#         parser = MMCIFParser(QUIET=True)
#         structure = parser.get_structure("structure", cif_path)
#         io = PDBIO()
#         io.set_structure(structure)
#         io.save(pdb_path)
#         if os.path.exists(pdb_path):
#             os.remove(cif_path) # 转换成功后删除 cif
#             return True
#     except Exception as e:
#         print(f"❌ CIF 转换失败 {cif_path}: {e}")
#         return False

# def download_and_process(pdb_id: str, assembly_num: str, save_dir: str):
#     pdb_id_upper = pdb_id.upper()
#     # 目标文件名统一为 .pdb{n}
#     final_pdb_name = f"{pdb_id_upper}.pdb{assembly_num}"
#     final_pdb_path = os.path.join(save_dir, final_pdb_name)

#     if os.path.exists(final_pdb_path):
#         return

#     # 1. 尝试直接下载 PDB
#     pdb_url = RCSB_PDB_URL.format(pdb_id_upper, assembly_num)
#     gz_path = final_pdb_path + ".gz"
    
#     try:
#         r = requests.get(pdb_url, timeout=15)
#         if r.status_code == 200:
#             with open(gz_path, "wb") as f:
#                 f.write(r.content)
#             with gzip.open(gz_path, "rb") as f_in, open(final_pdb_path, "wb") as f_out:
#                 shutil.copyfileobj(f_in, f_out)
#             os.remove(gz_path)
#             print(f"✅ Downloaded PDB: {final_pdb_name}")
#             return
#     except:
#         pass

#     # 2. 如果 PDB 不存在，下载 CIF 并转换
#     cif_url = RCSB_CIF_URL.format(pdb_id_upper, assembly_num)
#     tmp_cif_path = os.path.join(save_dir, f"{pdb_id_upper}_{assembly_num}_tmp.cif")
#     cif_gz_path = tmp_cif_path + ".gz"

#     try:
#         r = requests.get(cif_url, timeout=15)
#         if r.status_code == 200:
#             with open(cif_gz_path, "wb") as f:
#                 f.write(r.content)
#             with gzip.open(cif_gz_path, "rb") as f_in, open(tmp_cif_path, "wb") as f_out:
#                 shutil.copyfileobj(f_in, f_out)
#             os.remove(cif_gz_path)
            
#             # 转换
#             if cif_to_pdb(tmp_cif_path, final_pdb_path):
#                 print(f"🔄 Downloaded CIF & Converted to: {final_pdb_name}")
#         else:
#             print(f"❌ Not Found: {pdb_id_upper} Assembly {assembly_num}")
#     except Exception as e:
#         print(f"⚠️ Error: {pdb_id_upper} -> {e}")

# # --- 任务收集逻辑 ---
# download_tasks = set()
# for query_name in os.listdir(report_root):
#     query_dir = os.path.join(report_root, query_name)
#     if not os.path.isdir(query_dir): continue
#     for file_name in os.listdir(query_dir):
#         if not file_name.endswith(".m8"): continue
#         with open(os.path.join(query_dir, file_name), "r") as f:
#             for line in f:
#                 parts = line.strip().split("\t")
#                 if len(parts) < 2 or "-assembly" not in parts[1]: continue
#                 try:
#                     p_id = parts[1].split("-")[0]
#                     a_num = parts[1].split("-")[1].replace("assembly", "").split(".")[0]
#                     if not os.path.exists(os.path.join(output_root, f"{p_id.upper()}.pdb{a_num}")):
#                         download_tasks.add((p_id, a_num, output_root))
#                 except: continue

# # --- 并行执行 ---
# if download_tasks:
#     print(f"🚀 开始处理 {len(download_tasks)} 个模板任务...")
#     with ThreadPoolExecutor(max_workers=32) as executor:
#         futures = [executor.submit(download_and_process, p, a, o) for p, a, o in download_tasks]
#         for _ in as_completed(futures): pass
#     print("\n✅ 所有模板下载与转换完成。")