import argparse
import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser, is_aa
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TM-align for monomer templates and merge residue mappings."
    )
    parser.add_argument("--root", type=str, help="Monomer project root directory")
    args, _ = parser.parse_known_args()
    root_path = args.root if args.root else (sys.argv[1] if len(sys.argv) > 1 else None)
    if not root_path:
        print("❌ 错误: 请提供项目 root 路径")
        sys.exit(1)
    return os.path.abspath(root_path)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TMALIGN_BIN = os.path.join(SCRIPT_DIR, "TMalign")


def find_all_pdbs(directory):
    pdb_files = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".pdb"):
                pdb_files.append(os.path.join(root, file_name))
    return sorted(pdb_files)


def load_ranked_template_names(m8_path):
    if not os.path.exists(m8_path):
        return []

    ranked_names = []
    seen = set()
    with open(m8_path, "r") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            template_name = parts[1]
            if template_name in seen:
                continue
            ranked_names.append(template_name)
            seen.add(template_name)
    return ranked_names


def residue_token(residue):
    seq_id = residue.id[1]
    insertion_code = residue.id[2].strip()
    return f"{seq_id}{insertion_code}" if insertion_code else str(seq_id)


def residue_letter(residue):
    return protein_letters_3to1.get(residue.resname.strip().title(), "X")


def extract_protein_residues(pdb_path):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("query", pdb_path)
    except Exception:
        return []

    model = next(structure.get_models(), None)
    if model is None:
        return []

    for chain in model:
        residues = []
        for residue in chain:
            if residue.id[0] != " " or not is_aa(residue, standard=False):
                continue
            residues.append((residue_token(residue), residue_letter(residue)))
        if residues:
            return residues
    return []


def run_tmalign(query_pdb, target_pdb):
    if not os.path.exists(query_pdb) or not os.path.exists(target_pdb):
        return ""

    try:
        result = subprocess.run(
            [TMALIGN_BIN, query_pdb, target_pdb],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        return ""

    return result.stdout if result.returncode == 0 or result.stdout else ""


def parse_tmalign_output(output, tm_threshold=0.3):
    if not output:
        return None

    lines = output.splitlines()
    tm_score = None
    for line in lines:
        if "TM-score=" not in line:
            continue
        if "Chain_1" not in line and "Structure_1" not in line:
            continue
        match = re.search(r"TM-score=\s*([0-9.]+)", line)
        if match:
            tm_score = float(match.group(1))
            break

    if tm_score is None or tm_score < tm_threshold:
        return None

    seq_start = -1
    for index, line in enumerate(lines):
        if line.startswith('(":" denotes residue pairs'):
            seq_start = index + 1
            break

    if seq_start == -1 or seq_start + 2 >= len(lines):
        return None

    query_seq = lines[seq_start].strip().upper()
    target_seq = lines[seq_start + 2].strip().upper()
    return query_seq, target_seq


def build_alignment_rows(query_seq, target_seq, query_residues, target_residues):
    query_index = 0
    target_index = 0
    rows = []

    for q_char, t_char in zip(query_seq, target_seq):
        if q_char == "*" or t_char == "*":
            continue

        query_residue = None
        target_residue = None

        if q_char != "-":
            if query_index >= len(query_residues):
                return None
            query_residue = query_residues[query_index]
            query_index += 1

        if t_char != "-":
            if target_index >= len(target_residues):
                return None
            target_residue = target_residues[target_index]
            target_index += 1

        if query_residue is None:
            continue

        target_token = target_residue[0] if target_residue else "-"
        target_letter = target_residue[1] if target_residue else "-"
        rows.append((query_residue[0], query_residue[1], target_letter, target_token))

    return rows if rows else None


def process_single_query(query_name, root_dirs):
    query_parent_dir = os.path.join(root_dirs["query"], query_name)
    if not os.path.exists(query_parent_dir):
        return f"ERR: Missing query dir {query_name}"

    query_pdbs = find_all_pdbs(query_parent_dir)
    if not query_pdbs:
        return f"ERR: No PDB found for query {query_name}"
    query_pdb = query_pdbs[0]

    query_residues = extract_protein_residues(query_pdb)
    if not query_residues:
        return f"ERR: No protein residues found for query {query_name}"

    target_parent_dir = os.path.join(root_dirs["target"], query_name)
    if not os.path.exists(target_parent_dir):
        return f"ERR: Missing target dir {query_name}"

    target_pdbs = find_all_pdbs(target_parent_dir)
    if not target_pdbs:
        return f"ERR: No template PDBs found in {target_parent_dir}"

    ranked_template_names = load_ranked_template_names(
        os.path.join(root_dirs["m8"], query_name, f"{query_name}_results.m8")
    )
    template_path_map = {
        os.path.splitext(os.path.basename(target_pdb))[0]: target_pdb
        for target_pdb in target_pdbs
    }
    ordered_template_names = [name for name in ranked_template_names if name in template_path_map]
    ordered_template_names.extend(
        sorted(name for name in template_path_map if name not in set(ordered_template_names))
    )
    target_pdbs = [template_path_map[name] for name in ordered_template_names]

    row_order = [residue_id for residue_id, _ in query_residues]
    query_seq_map = {residue_id: residue_aa for residue_id, residue_aa in query_residues}
    row_dict = {residue_id: {} for residue_id in row_order}
    aligned_template_rows = {}

    def task(target_pdb):
        target_name = os.path.splitext(os.path.basename(target_pdb))[0]
        output = run_tmalign(query_pdb, target_pdb)
        parsed = parse_tmalign_output(output)
        if not parsed:
            return target_name, None

        target_residues = extract_protein_residues(target_pdb)
        if not target_residues:
            return target_name, None

        alignment_rows = build_alignment_rows(
            query_seq=parsed[0],
            target_seq=parsed[1],
            query_residues=query_residues,
            target_residues=target_residues,
        )
        return target_name, alignment_rows

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(task, target_pdb) for target_pdb in target_pdbs]
        for future in as_completed(futures):
            target_name, alignment_rows = future.result()
            if not alignment_rows:
                continue

            aligned_template_rows[target_name] = alignment_rows

    if not aligned_template_rows:
        return f"SKIP: {query_name} (No templates passed TM-score threshold)"

    output_dir = os.path.join(root_dirs["output"], query_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{query_name}_alignment.txt")

    # Foldseek stage already enforces top-N ranking. Keep that retained pool order here.
    target_names = [name for name in ordered_template_names if name in aligned_template_rows]
    for target_name in target_names:
        target_map = {
            query_res_id: (target_aa, target_res_id)
            for query_res_id, _, target_aa, target_res_id in aligned_template_rows[target_name]
        }
        for residue_id in row_order:
            row_dict[residue_id][target_name] = target_map.get(residue_id, ("-", "-"))

    header = ["query_seq", "query_res_num"]
    for name in target_names:
        header.extend([f"{name}_seq", f"{name}_res_num"])

    with open(output_file, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        for residue_id in row_order:
            row = [query_seq_map[residue_id], residue_id]
            for name in target_names:
                row.extend(row_dict[residue_id].get(name, ("-", "-")))
            writer.writerow(row)

    return f"OK: {query_name} (Aligned {len(target_names)} templates)"


def main():
    root = parse_args()
    root_dirs = {
        "query": os.path.join(root, "pdb"),
        "target": os.path.join(root, "templates_cat"),
        "m8": os.path.join(root, "search_result", "result"),
        "output": os.path.join(root, "align_results"),
    }

    if not os.path.exists(TMALIGN_BIN):
        print(f"❌ 错误: 未找到 {TMALIGN_BIN}")
        sys.exit(1)

    all_queries = sorted(
        directory
        for directory in os.listdir(root_dirs["query"])
        if os.path.isdir(os.path.join(root_dirs["query"], directory))
    )

    print(f"🚀 启动比对逻辑。Query 数量: {len(all_queries)}")

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_single_query, query, root_dirs): query for query in all_queries}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Mapping"):
            print(f"\n{future.result()}")


if __name__ == "__main__":
    main()
