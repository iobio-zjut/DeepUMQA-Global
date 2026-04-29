import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from Bio.PDB import PDBParser, is_aa
from scipy.spatial.distance import pdist, squareform


def parse_args():
    parser = argparse.ArgumentParser(description="Map monomer distance matrices from TM-align mappings.")
    parser.add_argument("--root", type=str, help="Monomer project root directory")
    args, _ = parser.parse_known_args()
    root_path = args.root if args.root else (sys.argv[1] if len(sys.argv) > 1 else None)
    if not root_path:
        print("❌ 错误: 请提供项目 root 路径")
        sys.exit(1)
    return os.path.abspath(root_path)


def residue_token(residue):
    seq_id = residue.id[1]
    insertion_code = residue.id[2].strip()
    return f"{seq_id}{insertion_code}" if insertion_code else str(seq_id)


def find_pdb_recursive(directory, target_id):
    base_id = target_id.replace(".pdb", "")
    for root, _, files in os.walk(directory):
        for file_name in sorted(files):
            if file_name == f"{base_id}.pdb" or file_name == f"{target_id}.pdb":
                return os.path.join(root, file_name)
    return None


def parse_alignment_file(align_path):
    try:
        with open(align_path, "r") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader)
            data = [row for row in reader]

        q_num_idx = next((i for i, column in enumerate(header) if column in ["query_res_num", "quary_res_num"]), None)
        if q_num_idx is None:
            return None, None

        query_res_ids = [row[q_num_idx] for row in data]
        target_info = {}
        for index, column in enumerate(header):
            if column.endswith("_res_num") and "query" not in column and "quary" not in column:
                target_name = column.replace("_res_num", "")
                res_ids = [None if row[index] in {"-", ""} else row[index] for row in data]
                target_info[target_name] = res_ids

        return query_res_ids, target_info
    except Exception as exc:
        print(f"  [!] Parse error in {align_path}: {exc}")
        return None, None


def extract_coords(pdb_file, residue_ids):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("", pdb_file)
        model = next(structure.get_models(), None)
        if model is None:
            return None
        chain = next((chain for chain in model if any(res.id[0] == " " and is_aa(res, standard=False) for res in chain)), None)
        if chain is None:
            return None

        residue_map = {
            residue_token(residue): residue
            for residue in chain.get_residues()
            if residue.id[0] == " " and is_aa(residue, standard=False)
        }

        coords = []
        for residue_id in residue_ids:
            if not residue_id or residue_id not in residue_map:
                coords.append([np.nan, np.nan, np.nan])
                continue

            residue = residue_map[residue_id]
            atom = residue["CB"] if "CB" in residue else (residue["CA"] if "CA" in residue else None)
            coords.append(atom.get_coord() if atom else [np.nan, np.nan, np.nan])
        return np.array(coords, dtype=np.float32)
    except Exception:
        return None


def calc_distance_matrix(coords):
    length = len(coords)
    dist_matrix = np.full((length, length), np.nan, dtype=np.float32)
    mask = ~np.isnan(coords).any(axis=1)
    valid_coords = coords[mask]
    if len(valid_coords) > 1:
        dists = squareform(pdist(valid_coords)).astype(np.float32)
        valid_indices = np.where(mask)[0]
        for i, index in enumerate(valid_indices):
            dist_matrix[index, valid_indices] = dists[i]
    return dist_matrix


def process_one_query(query_name, dirs):
    align_file = os.path.join(dirs["align"], query_name, f"{query_name}_alignment.txt")
    if not os.path.exists(align_file):
        return f"SKIP | {query_name}: No alignment file at {align_file}"

    query_res_ids, target_info = parse_alignment_file(align_file)
    if query_res_ids is None:
        return f"FAIL | {query_name}: Could not parse alignment"

    output_dir = os.path.join(dirs["output"], query_name)
    os.makedirs(output_dir, exist_ok=True)

    query_pdb = find_pdb_recursive(os.path.join(dirs["query"], query_name), query_name)
    if query_pdb:
        query_coords = extract_coords(query_pdb, query_res_ids)
        if query_coords is not None:
            query_dist = calc_distance_matrix(query_coords)
            np.save(os.path.join(output_dir, f"{query_name}_q.npy"), query_dist)

    success_count = 0
    target_search_base = os.path.join(dirs["target"], query_name)
    for target_name, residue_ids in target_info.items():
        target_pdb = find_pdb_recursive(target_search_base, target_name)
        if not target_pdb:
            continue

        coords = extract_coords(target_pdb, residue_ids)
        if coords is None:
            continue

        dist_mat = calc_distance_matrix(coords)
        np.save(os.path.join(output_dir, f"{target_name}_t.npy"), dist_mat)
        success_count += 1

    return f"DONE | {query_name}: {success_count} templates mapped to {output_dir}"


def main():
    root = parse_args()
    dirs = {
        "query": os.path.join(root, "pdb"),
        "target": os.path.join(root, "templates_cat"),
        "align": os.path.join(root, "align_results"),
        "output": os.path.join(root, "CCdist"),
    }

    print(f"DEBUG | Looking for alignments in: {dirs['align']}")
    if not os.path.exists(dirs["align"]):
        print(f"❌ 错误: 比对结果目录不存在 {dirs['align']}")
        return

    all_queries = sorted(
        directory
        for directory in os.listdir(dirs["query"])
        if os.path.isdir(os.path.join(dirs["query"], directory))
    )
    print(f"🚀 开始映射单体距离矩阵。找到 {len(all_queries)} 个任务...")

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_one_query, query, dirs) for query in all_queries]
        for future in as_completed(futures):
            try:
                print(future.result())
            except Exception as exc:
                print(f"❌ 进程执行异常: {exc}")


if __name__ == "__main__":
    main()
