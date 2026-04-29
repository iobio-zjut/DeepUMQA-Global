import os
import sys
import subprocess
import re
import glob
import argparse
from Bio.PDB import PDBParser, MMCIFParser, is_aa
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 动态获取脚本目录和 USalign 路径 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_USALIGN = os.path.join(SCRIPT_DIR, "USalign")

# --- 参数解析 ---
def parse_args():
    parser = argparse.ArgumentParser(description="Structural alignment and residue mapping using USalign.")
    parser.add_argument('--root', type=str, help='Project root directory')
    # 增加 usalign 路径参数，默认为脚本同级目录下的文件
    parser.add_argument('--usalign', type=str, default=DEFAULT_USALIGN, help='Path to USalign executable')
    args, unknown = parser.parse_known_args()
    
    root_path = args.root if args.root else (sys.argv[1] if len(sys.argv) > 1 else None)
    if not root_path:
        print("❌ 错误: 请提供项目 root 路径")
        sys.exit(1)
    
    # 检查 USalign 是否存在且可执行
    if not os.path.exists(args.usalign):
        print(f"❌ 错误: 未在 {args.usalign} 找到 USalign 执行文件")
        sys.exit(1)
        
    return os.path.abspath(root_path), args.usalign

# --- 结构解析工具 ---
def residue_token(residue):
    seq_id = residue.id[1]
    insertion_code = residue.id[2].strip()
    return f"{seq_id}{insertion_code}" if insertion_code else str(seq_id)


def extract_residues(structure_path, chain_order=None):
    """提取残基编号和链ID，自适应 PDB/CIF 格式"""
    ext = os.path.splitext(structure_path)[-1].lower()
    parser = MMCIFParser(QUIET=True) if ext == ".cif" else PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure('X', structure_path)
        residues = []
        for model in structure:
            ordered_chains = []
            seen = set()
            if chain_order:
                for chain_id in chain_order:
                    if chain_id in model and chain_id not in seen:
                        ordered_chains.append(model[chain_id])
                        seen.add(chain_id)
            for chain in model:
                if chain.id not in seen:
                    ordered_chains.append(chain)
                    seen.add(chain.id)

            for chain in ordered_chains:
                for res in chain:
                    if res.id[0] == ' ' and is_aa(res, standard=False):
                        residues.append((residue_token(res), chain.id))
            break 
        return residues
    except Exception:
        return []


def extract_chain_order(stdout_lines, structure_label, structure_path):
    prefix = f"Name of {structure_label}:"
    for line in stdout_lines:
        if not line.startswith(prefix):
            continue
        parts = line.split()
        if len(parts) < 4:
            break
        path_with_order = parts[3]
        if not path_with_order.startswith(structure_path):
            break
        suffix = path_with_order[len(structure_path):]
        return [token for token in suffix.split(":") if token]
    return None

def run_usalign(query_path, target_path, usalign_path, output_txt):
    """运行 USalign 并解析比对映射"""
    cmd = f"{usalign_path} {query_path} {target_path} -mm 1 -ter 1"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout_lines = result.stdout.splitlines()

    # 1. 提取 TM-score
    tm_score = None
    for line in stdout_lines:
        if "normalized by length of Structure_1" in line:
            try:
                tm_score = float(line.strip().split('=')[1].split()[0])
            except: pass
            break
    
    if tm_score is not None and tm_score < 0.15:
        return f"SKIP (TM={tm_score:.3f})"

    # 2. 提取对齐序列
    aln_q, aln_t = None, None
    seq_start = -1
    for i, line in enumerate(stdout_lines):
        if line.startswith('(":" denotes residue pairs'):
            seq_start = i + 1
            break

    if seq_start != -1 and seq_start + 2 < len(stdout_lines):
        aln_q = stdout_lines[seq_start].strip()
        aln_t = stdout_lines[seq_start + 2].strip()

    if not aln_q or not aln_t:
        return "ERR_NO_ALN"

    # 3. 映射残基
    query_chain_order = extract_chain_order(stdout_lines, "Structure_1", query_path)
    target_chain_order = extract_chain_order(stdout_lines, "Structure_2", target_path)
    query_res_list = extract_residues(query_path, query_chain_order)
    target_res_list = extract_residues(target_path, target_chain_order)
    
    q_base = os.path.basename(query_path).replace(".pdb", "")
    t_base = re.sub(r"\.cif$", "", os.path.basename(target_path))
    t_base = re.sub(r"\.pdb\d*", "", t_base)
    
    q_idx, t_idx = 0, 0
    output_lines = []
    for i in range(len(aln_q)):
        q_char, t_char = aln_q[i], aln_t[i]
        # USalign uses "*" as a chain separator for multimers. It is not a residue.
        if q_char == "*" or t_char == "*":
            continue
        q_resnum, q_chain = (query_res_list[q_idx] if q_idx < len(query_res_list) else ("!", "!")) if q_char != '-' else ("-", "-")
        if q_char != '-': q_idx += 1
        t_resnum, t_chain = (target_res_list[t_idx] if t_idx < len(target_res_list) else ("!", "!")) if t_char != '-' else ("-", "-")
        if t_char != '-': t_idx += 1
        if q_resnum != '-': 
            output_lines.append(f"{q_base} {q_resnum} {q_chain} {t_base} {t_resnum} {t_chain}")

    if output_lines:
        os.makedirs(os.path.dirname(output_txt), exist_ok=True)
        with open(output_txt, 'w') as f:
            f.write("query_name res_num chain_id target_name res_num chain_id\n")
            f.writelines("\n".join(output_lines) + "\n")
        return f"SUCCESS (TM={tm_score:.3f})"
    return "ERR_EMPTY_MAP"

# --- 任务分发逻辑 ---
def process_single_line(line, subdir, q_cat_root, t_cat_root, usalign_path, output_dir):
    parts = line.strip().split()
    if len(parts) < 4: return "FMT_ERR"
    q_id, t_raw_id, q_chains_str, t_chains_str = parts[0], parts[1], parts[2], parts[3]
    q_chains_fn, t_chains_fn = q_chains_str.replace(',', '_'), t_chains_str.replace(',', '_')

    q_file = f"{q_id}_{q_chains_fn}.pdb"
    q_path = os.path.join(q_cat_root, subdir, q_id, q_file)

    base_t = os.path.basename(t_raw_id).replace('.gz', '')
    pdb_id, asm_id = None, "1"
    if '-assembly' in base_t:
        pdb_id, rest = base_t.split('-assembly')
        asm_id = rest.split('.')[0]
    else:
        pdb_id = base_t.split('.')[0]
        asm_id = re.sub(r'[^0-9]', '', base_t.split('.')[-1]) if '.' in base_t else "1"

    t_dir = os.path.join(t_cat_root, subdir, q_id)
    t_prefix = f"{pdb_id.upper()}.pdb{asm_id}_{t_chains_fn}."
    found_t_path = None
    if os.path.exists(t_dir):
        for f in os.listdir(t_dir):
            if f.startswith(t_prefix):
                found_t_path = os.path.join(t_dir, f)
                break

    if not os.path.exists(q_path): return f"Q_MISSING | {q_id}"
    if not found_t_path: return f"T_MISSING | {pdb_id}_{t_chains_fn}"

    out_name = f"{q_id}_{q_chains_fn}_vs_{pdb_id.upper()}_{t_chains_fn}.txt"
    out_path = os.path.join(output_dir, subdir, q_id, out_name)
    
    if os.path.exists(out_path): return f"EXISTS | {out_name}"
    return f"{run_usalign(q_path, found_t_path, usalign_path, out_path)} | {out_name}"

if __name__ == "__main__":
    ROOT, usalign_exe = parse_args()
    
    search_dir = os.path.join(ROOT, "search_result", "result")
    q_cat_root = os.path.join(ROOT, "query_cat")
    t_cat_root = os.path.join(ROOT, "templates_cat")
    output_dir = os.path.join(ROOT, "align_results")

    print(f"📂 项目根目录: {ROOT}")
    print(f"🔧 使用 USalign: {usalign_exe}")

    report_files = glob.glob(os.path.join(search_dir, '**', '*.m8_report'), recursive=True)
    tasks = []
    for rf in report_files:
        subdir = os.path.basename(os.path.dirname(rf))
        try:
            with open(rf, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        tasks.append((line.strip(), subdir, q_cat_root, t_cat_root, usalign_exe, output_dir))
        except: continue

    total_tasks = len(tasks)
    if total_tasks > 0:
        with ProcessPoolExecutor(max_workers=min(40, os.cpu_count())) as executor:
            futures = [executor.submit(process_single_line, *t) for t in tasks]
            for i, f in enumerate(as_completed(futures), 1):
                try:
                    print(f"[{i}/{total_tasks}] {f.result()}")
                except Exception as e:
                    print(f"[{i}/{total_tasks}] 运行异常: {e}")

    print("✨ 所有比对映射任务执行完毕。")

