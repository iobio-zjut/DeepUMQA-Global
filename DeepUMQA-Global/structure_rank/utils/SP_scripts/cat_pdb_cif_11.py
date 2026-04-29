import os
import sys
import glob
import copy
import re
import argparse
from Bio.PDB import PDBParser, PDBIO, Model, Structure
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 参数解析 ---
def parse_args():
    parser = argparse.ArgumentParser(description="Extract specific chains from Query and Template structures.")
    parser.add_argument('--root', type=str, help='Project root directory')
    args, unknown = parser.parse_known_args()
    root_path = args.root if args.root else (sys.argv[1] if len(sys.argv) > 1 else None)
    if not root_path:
        print("❌ 错误: 请提供项目根目录路径")
        sys.exit(1)
    return os.path.abspath(root_path)

# --- 核心提取逻辑 ---

def extract_chains_robust(infile, chains, outfile):
    success = False
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("structure", infile)
        if len(structure) > 0:
            new_structure = Structure.Structure("new")
            model = Model.Model(0)
            found = False
            for chain_id in chains:
                for chain in structure[0]:
                    if chain.id == chain_id:
                        model.add(copy.deepcopy(chain))
                        found = True
                        break
            if found:
                new_structure.add(model)
                io = PDBIO()
                io.set_structure(new_structure)
                io.save(outfile)
                if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
                    success = True
    except:
        success = False

    # 文本流回退逻辑 (针对格式不规范的 PDB)
    if not success:
        try:
            with open(infile, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            output_lines = []
            chain_set = set(chains)
            for line in lines:
                if line.startswith(("ATOM", "HETATM")):
                    if line[21].strip() in chain_set:
                        output_lines.append(line)
            if output_lines:
                with open(outfile, 'w') as f:
                    f.writelines(output_lines)
                    f.write("END\n")
                success = True
        except:
            success = False
    return success

def extract_cif_chains_raw(infile, chains, outfile):
    try:
        mmcif_dict = MMCIF2Dict(infile)
        auth_asym_ids = mmcif_dict['_atom_site.auth_asym_id']
        with open(infile, 'r') as f:
            lines = f.readlines()
        atom_start = next((i for i, l in enumerate(lines) if l.strip().startswith("loop_") and "_atom_site." in lines[i+1]), None)
        if atom_start is None: return False
        
        atom_fields = []
        for l in lines[atom_start+1:]:
            if l.strip().startswith("_atom_site."):
                atom_fields.append(l)
            else:
                break
        
        atom_data_start = atom_start + 1 + len(atom_fields)
        data_lines = []
        for chain_id in chains:
            indices = [i for i, asym_id in enumerate(auth_asym_ids) if asym_id == chain_id]
            data_lines.extend([lines[atom_data_start + idx] for idx in indices])
        
        if not data_lines: return False
        with open(outfile, 'w') as fout:
            fout.writelines(lines[:atom_start])
            fout.write("loop_\n")
            fout.writelines(atom_fields)
            fout.writelines(data_lines)
        return True
    except:
        return False

# --- 任务分发逻辑 ---

def process_report_line(report_line, subdir, query_dir, target_dir, query_out_dir, target_out_dir):
    cols = report_line.strip().split()
    if len(cols) < 4: return f"SKIP: Line format error"
    
    query_raw_id, target_raw_id = cols[0], cols[1]
    query_chains = cols[2].split(',')
    target_chains = cols[3].split(',')

    # 1. Query 处理
    potential_names = [query_raw_id]
    if "_" in query_raw_id:
        potential_names.append("_".join(query_raw_id.split("_")[:-1]))

    found_q_in = None
    for name in potential_names:
        p = os.path.join(query_dir, subdir, f"{name}.pdb")
        if os.path.exists(p):
            found_q_in = p
            break

    q_status = "MISSING"
    if found_q_in:
        q_out_subdir = os.path.join(query_out_dir, subdir, query_raw_id)
        q_out_file = os.path.join(q_out_subdir, f"{query_raw_id}_{'_'.join(query_chains)}.pdb")
        
        if os.path.exists(q_out_file):
            q_status = "EXISTS"
        else:
            temp_q = q_out_file + ".tmp"
            os.makedirs(q_out_subdir, exist_ok=True)
            if extract_chains_robust(found_q_in, query_chains, temp_q):
                os.rename(temp_q, q_out_file)
                q_status = "SUCCESS"
            else:
                if os.path.exists(temp_q): os.remove(temp_q)
                q_status = "EXTRACT_FAIL"

    # 2. Target 处理 (模板)
    base_name = os.path.basename(target_raw_id).replace('.gz', '')
    pdb_id, asm_id = None, "1"
    if '-assembly' in base_name:
        pdb_id, rest = base_name.split('-assembly')
        asm_id = rest.split('.')[0]
    else:
        pdb_id = base_name.split('.')[0]
        asm_id = re.sub(r'[^0-9]', '', base_name.split('.')[-1]) if '.' in base_name else "1"

    found_t_in = None
    for t_fmt in [f"{pdb_id.upper()}.pdb{asm_id}", f"{pdb_id.upper()}-assembly{asm_id}.cif"]:
        t_p = os.path.join(target_dir, t_fmt)
        if os.path.exists(t_p):
            found_t_in = t_p
            break

    t_status = "MISSING"
    if found_t_in:
        t_out_subdir = os.path.join(target_out_dir, subdir, query_raw_id)
        ext = os.path.splitext(found_t_in)[-1].lower()
        t_out_file = os.path.join(t_out_subdir, f"{pdb_id.upper()}.pdb{asm_id}_{'_'.join(target_chains)}{ext}")
        
        if os.path.exists(t_out_file):
            t_status = "EXISTS"
        else:
            temp_t = t_out_file + ".tmp"
            os.makedirs(t_out_subdir, exist_ok=True)
            success = extract_cif_chains_raw(found_t_in, target_chains, temp_t) if ext == ".cif" else extract_chains_robust(found_t_in, target_chains, temp_t)
            if success:
                os.rename(temp_t, t_out_file)
                t_status = "SUCCESS"
            else:
                if os.path.exists(temp_t): os.remove(temp_t)
                t_status = "EXTRACT_FAIL"
    
    return f"Q:{q_status} | T:{t_status} | {query_raw_id}"

# --- 主程序 ---

if __name__ == "__main__":
    ROOT = parse_args()
    
    search_dir = os.path.join(ROOT, "search_result", "result")
    q_in = os.path.join(ROOT, "pdb")
    t_in = os.path.join(ROOT, "templates")
    q_out = os.path.join(ROOT, "query_cat")
    t_out = os.path.join(ROOT, "templates_cat")

    print(f"📂 项目根目录: {ROOT}")
    report_files = glob.glob(os.path.join(search_dir, '**', '*.m8_report'), recursive=True)
    
    tasks = []
    for rf in report_files:
        subdir = os.path.basename(os.path.dirname(rf))
        try:
            with open(rf, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        tasks.append((line.strip(), subdir, q_in, t_in, q_out, t_out))
        except:
            continue

    total = len(tasks)
    print(f"🚀 锁定 .m8_report: 共有 {total} 个链提取子任务...")

    if total > 0:
        with ProcessPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
            futures = [executor.submit(process_report_line, *t) for t in tasks]
            for i, f in enumerate(as_completed(futures), 1):
                if i % 100 == 0 or i == total:
                    try:
                        res = f.result()
                        print(f"  [进度 {i}/{total}] {res}")
                    except Exception as e:
                        print(f"  [错误] 任务 {i} 执行异常: {e}")

    print("✨ 全部链提取任务处理完成。")



# import os
# import sys
# import glob
# import copy
# import re
# from Bio.PDB import PDBParser, PDBIO, Model, Structure
# from Bio.PDB.MMCIF2Dict import MMCIF2Dict
# from concurrent.futures import ProcessPoolExecutor, as_completed

# # --- 核心提取逻辑 ---

# def extract_chains_robust(infile, chains, outfile):
#     success = False
#     try:
#         parser = PDBParser(QUIET=True)
#         structure = parser.get_structure("structure", infile)
#         if len(structure) > 0:
#             new_structure = Structure.Structure("new")
#             model = Model.Model(0)
#             found = False
#             for chain_id in chains:
#                 for chain in structure[0]:
#                     if chain.id == chain_id:
#                         model.add(copy.deepcopy(chain))
#                         found = True
#                         break
#             if found:
#                 new_structure.add(model)
#                 io = PDBIO()
#                 io.set_structure(new_structure)
#                 io.save(outfile)
#                 if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
#                     success = True
#     except:
#         success = False

#     # 文本流回退逻辑
#     if not success:
#         try:
#             with open(infile, 'r', encoding='utf-8', errors='ignore') as f:
#                 lines = f.readlines()
#             output_lines = []
#             chain_set = set(chains)
#             for line in lines:
#                 if line.startswith(("ATOM", "HETATM")):
#                     if line[21].strip() in chain_set:
#                         output_lines.append(line)
#             if output_lines:
#                 with open(outfile, 'w') as f:
#                     f.writelines(output_lines)
#                     f.write("END\n")
#                 success = True
#         except:
#             success = False
#     return success

# def extract_cif_chains_raw(infile, chains, outfile):
#     try:
#         mmcif_dict = MMCIF2Dict(infile)
#         auth_asym_ids = mmcif_dict['_atom_site.auth_asym_id']
#         with open(infile, 'r') as f:
#             lines = f.readlines()
#         atom_start = next((i for i, l in enumerate(lines) if l.strip().startswith("loop_") and "_atom_site." in lines[i+1]), None)
#         if atom_start is None: return False
        
#         # 提取字段定义行
#         atom_fields = []
#         for l in lines[atom_start+1:]:
#             if l.strip().startswith("_atom_site."):
#                 atom_fields.append(l)
#             else:
#                 break
        
#         atom_data_start = atom_start + 1 + len(atom_fields)
#         data_lines = []
#         for chain_id in chains:
#             indices = [i for i, asym_id in enumerate(auth_asym_ids) if asym_id == chain_id]
#             data_lines.extend([lines[atom_data_start + idx] for idx in indices])
        
#         if not data_lines: return False
#         with open(outfile, 'w') as fout:
#             fout.writelines(lines[:atom_start])
#             fout.write("loop_\n")
#             fout.writelines(atom_fields)
#             fout.writelines(data_lines)
#         return True
#     except:
#         return False

# # --- 任务分发逻辑 ---

# def process_report_line(report_line, subdir, query_dir, target_dir, query_out_dir, target_out_dir):
#     cols = report_line.strip().split()
#     if len(cols) < 4: return f"SKIP: Line format error"
    
#     query_raw_id, target_raw_id = cols[0], cols[1]
#     query_chains = cols[2].split(',')
#     target_chains = cols[3].split(',')

#     # 1. Query 处理
#     potential_names = [query_raw_id]
#     if "_" in query_raw_id:
#         potential_names.append("_".join(query_raw_id.split("_")[:-1]))

#     found_q_in = None
#     for name in potential_names:
#         p = os.path.join(query_dir, subdir, f"{name}.pdb")
#         if os.path.exists(p):
#             found_q_in = p
#             break

#     q_status = "MISSING"
#     if found_q_in:
#         q_out_subdir = os.path.join(query_out_dir, subdir, query_raw_id)
#         q_out_file = os.path.join(q_out_subdir, f"{query_raw_id}_{'_'.join(query_chains)}.pdb")
        
#         if os.path.exists(q_out_file):
#             q_status = "EXISTS"
#         else:
#             temp_q = q_out_file + ".tmp"
#             os.makedirs(q_out_subdir, exist_ok=True)
#             if extract_chains_robust(found_q_in, query_chains, temp_q):
#                 os.rename(temp_q, q_out_file)
#                 q_status = "SUCCESS"
#             else:
#                 if os.path.exists(temp_q): os.remove(temp_q)
#                 if not os.listdir(q_out_subdir): os.rmdir(q_out_subdir)
#                 q_status = "EXTRACT_FAIL"
#     else:
#         # Debug: 打印具体没找到的路径
#         print(f"  [DEBUG] Query PDB not found: {os.path.join(query_dir, subdir, query_raw_id)}.pdb")

#     # 2. Target 处理 (模板)
#     base_name = os.path.basename(target_raw_id).replace('.gz', '')
#     pdb_id, asm_id = None, "1"
#     if '-assembly' in base_name:
#         pdb_id, rest = base_name.split('-assembly')
#         asm_id = rest.split('.')[0]
#     else:
#         pdb_id = base_name.split('.')[0]
#         asm_id = re.sub(r'[^0-9]', '', base_name.split('.')[-1]) if '.' in base_name else "1"

#     found_t_in = None
#     for t_fmt in [f"{pdb_id.upper()}.pdb{asm_id}", f"{pdb_id.upper()}-assembly{asm_id}.cif"]:
#         t_p = os.path.join(target_dir, t_fmt)
#         if os.path.exists(t_p):
#             found_t_in = t_p
#             break

#     t_status = "MISSING"
#     if found_t_in:
#         t_out_subdir = os.path.join(target_out_dir, subdir, query_raw_id)
#         ext = os.path.splitext(found_t_in)[-1].lower()
#         t_out_file = os.path.join(t_out_subdir, f"{pdb_id.upper()}.pdb{asm_id}_{'_'.join(target_chains)}{ext}")
        
#         if os.path.exists(t_out_file):
#             t_status = "EXISTS"
#         else:
#             temp_t = t_out_file + ".tmp"
#             os.makedirs(t_out_subdir, exist_ok=True)
#             success = extract_cif_chains_raw(found_t_in, target_chains, temp_t) if ext == ".cif" else extract_chains_robust(found_t_in, target_chains, temp_t)
#             if success:
#                 os.rename(temp_t, t_out_file)
#                 t_status = "SUCCESS"
#             else:
#                 if os.path.exists(temp_t): os.remove(temp_t)
#                 if os.path.exists(t_out_subdir) and not os.listdir(t_out_subdir): os.rmdir(t_out_subdir)
#                 t_status = "EXTRACT_FAIL"
    
#     return f"Q:{q_status} | T:{t_status} | {query_raw_id}"

# # --- 主程序 ---

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python script.py <PROJECT_ROOT>")
#         sys.exit(1)

#     ROOT = sys.argv[1]
#     search_dir = os.path.join(ROOT, "search_result", "result")
#     q_in = os.path.join(ROOT, "pdb")
#     t_in = os.path.join(ROOT, "templates")
#     q_out = os.path.join(ROOT, "query_cat")
#     t_out = os.path.join(ROOT, "templates_cat")

#     report_files = glob.glob(os.path.join(search_dir, '**', '*.m8_report'), recursive=True)
    
#     tasks = []
#     for rf in report_files:
#         subdir = os.path.basename(os.path.dirname(rf))
#         with open(rf, 'r') as f:
#             for line in f:
#                 if line.strip() and not line.startswith("#"):
#                     tasks.append((line.strip(), subdir, q_in, t_in, q_out, t_out))

#     total = len(tasks)
#     print(f"🚀 锁定 .m8_report: 共有 {total} 个提取任务...")

#     with ProcessPoolExecutor(max_workers=32) as executor:
#         futures = [executor.submit(process_report_line, *t) for t in tasks]
#         for i, f in enumerate(as_completed(futures), 1):
#             # 每 100 次输出一次汇总进度
#             if i % 100 == 0 or i == total:
#                 res = f.result()
#                 print(f"  [进度 {i}/{total}] {res}")

#     print("✨ 全部链提取任务已完成。")