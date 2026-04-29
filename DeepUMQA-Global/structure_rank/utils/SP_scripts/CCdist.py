import os
import sys
import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, is_aa
from scipy.spatial.distance import pdist, squareform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project template coordinates onto the full query canvas."
    )
    parser.add_argument("--root", type=str, help="Project root directory")
    args, _ = parser.parse_known_args()

    root_path = args.root if args.root else (sys.argv[1] if len(sys.argv) > 1 else None)
    if not root_path:
        print("❌ 错误: 请提供项目 root 路径")
        sys.exit(1)
    return os.path.abspath(root_path)


pdb_parser = PDBParser(QUIET=True)
cif_parser = MMCIFParser(QUIET=True)


def residue_token(residue):
    seq_id = residue.id[1]
    insertion_code = residue.id[2].strip()
    return f"{seq_id}{insertion_code}" if insertion_code else str(seq_id)


def normalize_chain_tag(chain_tag):
    tokens = []
    for token in chain_tag.replace(",", "_").split("_"):
        if not token:
            continue
        tokens.append(re.sub(r"-\d+$", "", token))
    return "_".join(tokens)


def load_structure(structure_path):
    if not structure_path or not os.path.exists(structure_path):
        return None
    try:
        if ".cif" in structure_path.lower():
            return cif_parser.get_structure("", structure_path)
        return pdb_parser.get_structure("", structure_path)
    except Exception:
        return None


def build_residue_maps(structure):
    if structure is None:
        return [], {}

    ordered_residues = []
    residue_to_index = {}
    model = structure[0]

    for chain in model:
        for residue in chain:
            if residue.id[0] != " " or not is_aa(residue, standard=False):
                continue
            key = (chain.id, residue_token(residue))
            residue_to_index[key] = len(ordered_residues)
            ordered_residues.append((key, residue))

    return ordered_residues, residue_to_index


def build_target_residue_map(structure):
    residue_map = {}
    if structure is None:
        return residue_map

    model = structure[0]
    for chain in model:
        for residue in chain:
            if residue.id[0] != " " or not is_aa(residue, standard=False):
                continue
            residue_map[(chain.id, residue_token(residue))] = residue
    return residue_map


def get_residue_coord(residue):
    if residue is None:
        return None
    if "CB" in residue:
        return residue["CB"].get_coord()
    if "CA" in residue:
        return residue["CA"].get_coord()
    return None


def calc_distance_matrix(coords):
    length = len(coords)
    dist_matrix = np.full((length, length), np.nan, dtype=np.float32)
    mask = ~np.isnan(coords).any(axis=1)
    if np.sum(mask) > 1:
        valid_coords = coords[mask]
        dists = squareform(pdist(valid_coords)).astype(np.float32)
        valid_indices = np.where(mask)[0]
        for i, vi in enumerate(valid_indices):
            dist_matrix[vi, valid_indices] = dists[i]
    return dist_matrix


def parse_align_txt(txt_file):
    if not os.path.exists(txt_file):
        return None

    with open(txt_file, "r") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if len(lines) < 2:
        return None

    mappings = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 6:
            continue

        q_res_token, q_chain = parts[1], parts[2]
        t_res_token, t_chain = parts[4], parts[5]

        if q_res_token in {"-", "!"} or q_chain in {"-", "!"}:
            continue

        query_key = (q_chain, q_res_token)

        target_key = None
        if t_res_token not in {"-", "!"} and t_chain not in {"-", "!"}:
            target_key = (t_chain, t_res_token)

        mappings.append((query_key, target_key))

    return mappings if mappings else None


def get_target_pdb_path(target_base_dir, group, query_dir, mapping_filename):
    if "_vs_" not in mapping_filename:
        return None

    target_part = mapping_filename[:-4].split("_vs_")[1]
    target_id = target_part.split("_")[0]
    target_chains = normalize_chain_tag("_".join(target_part.split("_")[1:]))

    search_dir = os.path.join(target_base_dir, group, query_dir)
    if not os.path.isdir(search_dir):
        return None

    expected_prefix = f"{target_id.upper()}.PDB"
    for filename in os.listdir(search_dir):
        if not filename.upper().startswith(expected_prefix):
            continue
        match = re.match(r"^[^_]+_(.+)\.(?:pdb\d+|cif)$", filename)
        if not match:
            continue
        if normalize_chain_tag(match.group(1)) == target_chains:
            return os.path.join(search_dir, filename)
    return None


def build_query_aligned_target_coords(query_index_map, full_length, target_residue_map, mappings):
    coords = np.full((full_length, 3), np.nan, dtype=np.float32)
    for query_key, target_key in mappings:
        query_index = query_index_map.get(query_key)
        if query_index is None or target_key is None:
            continue

        target_residue = target_residue_map.get(target_key)
        coord = get_residue_coord(target_residue)
        if coord is not None:
            coords[query_index] = coord
    return coords


def process_one_align_file(txt_file, dirs):
    rel_path = os.path.relpath(txt_file, dirs["align"])
    parts = rel_path.split(os.sep)
    if len(parts) < 3:
        return f"FMT_ERR | {rel_path}"

    group, query_dir, filename = parts[0], parts[1], parts[2]
    out_dir = os.path.join(dirs["output"], group, query_dir)
    out_path = os.path.join(out_dir, f"{filename[:-4]}.npy")
    if os.path.exists(out_path):
        return f"SKIP | {filename}"

    full_query_pdb = os.path.join(dirs["pdb"], group, f"{query_dir}.pdb")
    target_pdb = get_target_pdb_path(dirs["target_cat"], group, query_dir, filename)
    if not os.path.exists(full_query_pdb) or not target_pdb:
        return f"MISSING | {filename}"

    mappings = parse_align_txt(txt_file)
    if not mappings:
        return f"FAIL_PARSE | {filename}"

    query_structure = load_structure(full_query_pdb)
    target_structure = load_structure(target_pdb)
    if query_structure is None or target_structure is None:
        return f"FAIL_STRUCT | {filename}"

    _, query_index_map = build_residue_maps(query_structure)
    target_residue_map = build_target_residue_map(target_structure)
    if not query_index_map:
        return f"FAIL_QUERY_INDEX | {filename}"

    target_coords = build_query_aligned_target_coords(
        query_index_map=query_index_map,
        full_length=len(query_index_map),
        target_residue_map=target_residue_map,
        mappings=mappings,
    )

    dist_matrix = calc_distance_matrix(target_coords)
    os.makedirs(out_dir, exist_ok=True)
    np.save(out_path, dist_matrix)
    return f"SUCCESS | {filename}"


def main():
    root = parse_args()
    dirs = {
        "pdb": os.path.join(root, "pdb"),
        "target_cat": os.path.join(root, "templates_cat"),
        "align": os.path.join(root, "align_results"),
        "output": os.path.join(root, "CCdist"),
    }

    tasks = []
    for current_root, _, files in os.walk(dirs["align"]):
        for filename in files:
            if filename.endswith(".txt") and "_vs_" in filename:
                tasks.append(os.path.join(current_root, filename))

    total = len(tasks)
    print(f"📂 项目根目录: {root}")
    print(f"🚀 任务启动: 找到 {total} 个比对文件，开始生成全长模板距离矩阵...")

    with ProcessPoolExecutor(max_workers=min(40, os.cpu_count())) as executor:
        futures = [executor.submit(process_one_align_file, task, dirs) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            if index % 100 == 0 or index == total:
                print(f"[{index}/{total}] {future.result()}")


if __name__ == "__main__":
    main()
