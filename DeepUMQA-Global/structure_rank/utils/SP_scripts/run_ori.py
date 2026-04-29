import argparse
import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
import scipy.spatial
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate orientation features and save merged NPZ files."
    )
    parser.add_argument(
        "--root",
        type=str,
        help="Legacy project root mode: use root/pdb as inputs and root/Profiles as both input/output.",
    )
    parser.add_argument(
        "--pdb_dir",
        type=str,
        help="Directory containing per-target PDB folders or PDB files.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing source NPZ files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory used to write merged orientation NPZ files. Defaults to input_dir.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="Optional TSV manifest: sample_name<TAB>pdb_path<TAB>input_npz_path<TAB>output_npz_path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(cpu_count(), 64),
        help="Worker count for multiprocessing.",
    )
    return parser.parse_args()


def build_virtual_cb(N, CA, C):
    b1 = N - CA
    b2 = C - CA
    b1 /= np.linalg.norm(b1)
    b2 /= np.linalg.norm(b2)
    n = np.cross(b1, b2)
    n /= np.linalg.norm(n)
    cb = CA + (-0.58273431 * b1) + (0.56802827 * b2) + (0.54067466 * n)
    return cb.astype(np.float32)


def parse_pdb_fast(pdb_path):
    coords = []
    atoms_needed = {"CA", "CB", "N", "C"}
    atom_dict = {}
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                if atom_name not in atoms_needed:
                    continue
                chain_id = line[21]
                resseq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atom_dict[(chain_id, resseq, atom_name)] = np.array([x, y, z], dtype=np.float32)

    residues = sorted(set((c, r) for c, r, a in atom_dict.keys()))
    for (chain_id, resseq) in residues:
        if (chain_id, resseq, "CA") in atom_dict and (chain_id, resseq, "N") in atom_dict:
            Ca = atom_dict[(chain_id, resseq, "CA")]
            N = atom_dict[(chain_id, resseq, "N")]
            if (chain_id, resseq, "CB") in atom_dict:
                Cb = atom_dict[(chain_id, resseq, "CB")]
            elif (chain_id, resseq, "C") in atom_dict:
                C = atom_dict[(chain_id, resseq, "C")]
                Cb = build_virtual_cb(N, Ca, C)
            else:
                continue
            coords.append((N, Ca, Cb))
    return coords


def get_dihedrals(a, b, c, d):
    b0 = -1.0 * (b - a)
    b1 = c - b
    b2 = d - c
    b1 /= np.linalg.norm(b1, axis=-1)[:, None]
    v = b0 - np.sum(b0 * b1, axis=-1)[:, None] * b1
    w = b2 - np.sum(b2 * b1, axis=-1)[:, None] * b1
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)
    return np.arctan2(y, x)


def get_angles(a, b, c):
    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:, None]
    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:, None]
    x = np.sum(v * w, axis=1)
    return np.arccos(np.clip(x, -1.0, 1.0))


def extract_orientation_features(coords, total_L):
    coords_arr = np.array(coords)
    N = coords_arr[:, 0, :]
    Ca = coords_arr[:, 1, :]
    Cb = coords_arr[:, 2, :]

    kdCb = scipy.spatial.cKDTree(Cb)
    pairs = kdCb.query_ball_tree(kdCb, r=20.0)
    idx_list = [(i, j) for i, js in enumerate(pairs) for j in js if i != j]

    omega6d = np.zeros((total_L, total_L), dtype=np.float32)
    theta6d = np.zeros((total_L, total_L), dtype=np.float32)
    phi6d = np.zeros((total_L, total_L), dtype=np.float32)

    if idx_list:
        idx0, idx1 = zip(*idx_list)
        idx0, idx1 = np.array(idx0), np.array(idx1)
        omega6d[idx0, idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])
        theta6d[idx0, idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])
        phi6d[idx0, idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])

    stack = np.stack([omega6d, theta6d, phi6d], axis=-1)
    return np.concatenate([np.sin(stack), np.cos(stack)], axis=-1)


def load_manifest(manifest_path):
    tasks = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid manifest row at line {line_no}: expected 4 tab-separated columns."
                )
            target_name, pdb_path, input_npz_path, output_npz_path = parts
            tasks.append(
                (
                    target_name,
                    os.path.abspath(pdb_path),
                    os.path.abspath(input_npz_path),
                    os.path.abspath(output_npz_path),
                )
            )
    return tasks


def discover_tasks(pdb_dir, input_dir, output_dir):
    tasks = []
    if not os.path.exists(pdb_dir):
        return tasks

    for target_name in sorted(os.listdir(pdb_dir)):
        target_path = os.path.join(pdb_dir, target_name)
        pdb_path = None

        if os.path.isdir(target_path):
            pdb_files = sorted(f for f in os.listdir(target_path) if f.endswith(".pdb"))
            if pdb_files:
                pdb_path = os.path.join(target_path, pdb_files[0])
        elif os.path.isfile(target_path) and target_path.endswith(".pdb"):
            pdb_path = target_path
            target_name = os.path.splitext(os.path.basename(target_path))[0]

        if not pdb_path:
            continue

        input_npz_path = os.path.join(input_dir, f"{target_name}.npz")
        output_npz_path = os.path.join(output_dir, f"{target_name}.npz")
        tasks.append((target_name, os.path.abspath(pdb_path), input_npz_path, output_npz_path))
    return tasks


def process_single_target(task_args):
    target_name, pdb_path, input_npz_path, output_npz_path = task_args

    if not os.path.exists(input_npz_path):
        return f"SKIP | {target_name} (input npz not found)"
    if not os.path.exists(pdb_path):
        return f"SKIP | {target_name} (pdb not found)"

    try:
        same_output = os.path.abspath(input_npz_path) == os.path.abspath(output_npz_path)
        if same_output and os.path.exists(output_npz_path):
            with np.load(output_npz_path, allow_pickle=True) as existing_out:
                if "orientation" in existing_out:
                    return f"EXISTS | {target_name}"

        with np.load(input_npz_path, allow_pickle=True) as data:
            existing_data = {k: data[k] for k in data.files}
            total_L = existing_data["mask"].shape[0]

        coords = parse_pdb_fast(pdb_path)
        if not coords:
            return f"FAIL | {target_name} (No valid residues)"

        orientation = extract_orientation_features(coords, total_L)
        existing_data["orientation"] = orientation

        os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
        np.savez_compressed(output_npz_path, **existing_data)
    except Exception as e:
        return f"ERROR | {target_name}: {str(e)}"

    return f"DONE | {target_name}"


def resolve_tasks(args):
    if args.manifest:
        return load_manifest(os.path.abspath(args.manifest))

    if args.root:
        root_path = os.path.abspath(args.root)
        pdb_dir = os.path.join(root_path, "pdb")
        input_dir = os.path.join(root_path, "Profiles")
        output_dir = os.path.abspath(args.output_dir) if args.output_dir else input_dir
        return discover_tasks(pdb_dir, input_dir, output_dir)

    if args.pdb_dir and args.input_dir:
        pdb_dir = os.path.abspath(args.pdb_dir)
        input_dir = os.path.abspath(args.input_dir)
        output_dir = os.path.abspath(args.output_dir) if args.output_dir else input_dir
        return discover_tasks(pdb_dir, input_dir, output_dir)

    print("❌ 错误: 请提供 --manifest，或 --root，或同时提供 --pdb_dir 与 --input_dir")
    sys.exit(1)


def main():
    args = parse_args()
    tasks = resolve_tasks(args)

    if not tasks:
        print("❌ 未发现待处理的任务。")
        sys.exit(0)

    print(f"🚀 发现 {len(tasks)} 个任务。开始提取 Orientation 特征...")

    worker_count = max(1, min(args.workers, len(tasks)))
    with Pool(processes=worker_count) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_target, tasks), total=len(tasks)))

    done = [r for r in results if r.startswith("DONE")]
    exists = [r for r in results if r.startswith("EXISTS")]
    failed = [r for r in results if r.startswith("FAIL") or r.startswith("ERROR")]
    skipped = [r for r in results if r.startswith("SKIP")]

    print(
        f"✨ 完成: {len(done)} 新生成, {len(exists)} 已存在, "
        f"{len(skipped)} 跳过, {len(failed)} 失败。"
    )

    if failed:
        for item in failed[:20]:
            print(item)
        sys.exit(1)


if __name__ == "__main__":
    main()
