import os
import sys
import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
from Bio.PDB import PDBParser, is_aa


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate complex structural profiles from full-length distance matrices."
    )
    parser.add_argument("--root", type=str, help="Project root directory")
    args, _ = parser.parse_known_args()

    root_path = args.root if args.root else (sys.argv[1] if len(sys.argv) > 1 else None)
    if not root_path:
        print("❌ 错误: 请提供项目 root 路径")
        sys.exit(1)
    return os.path.abspath(root_path)


def get_pdb_full_info(pdb_path):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("query", pdb_path)
        total_residues = 0
        for chain in structure[0]:
            residues = [res for res in chain if res.id[0] == " " and is_aa(res, standard=False)]
            total_residues += len(residues)
        return total_residues
    except Exception as exc:
        print(f"[ERROR] PDB parsing failed: {pdb_path}, {exc}")
        return 0


def load_all_matrices(cc_dir, q_cc_dir, group, complex_id, total_length):
    matrices = []

    template_dir = os.path.join(cc_dir, group, complex_id)
    if os.path.isdir(template_dir):
        for filename in sorted(os.listdir(template_dir)):
            if not filename.endswith(".npy"):
                continue
            file_path = os.path.join(template_dir, filename)
            try:
                matrix = np.load(file_path, mmap_mode="r")
                if matrix.ndim == 2 and matrix.shape == (total_length, total_length):
                    matrices.append(np.array(matrix, dtype=np.float32))
            except Exception:
                continue

    query_candidates = [
        os.path.join(q_cc_dir, group, complex_id, f"{complex_id}.npy"),
        os.path.join(q_cc_dir, group, f"{complex_id}.npy"),
    ]
    for file_path in query_candidates:
        if not os.path.isfile(file_path):
            continue
        try:
            matrix = np.load(file_path, mmap_mode="r")
            if matrix.ndim == 2 and matrix.shape == (total_length, total_length):
                matrices.append(np.array(matrix, dtype=np.float32))
                break
        except Exception:
            continue

    return matrices


def compute_structural_profile(matrices):
    if not matrices:
        return None, None, None

    length = matrices[0].shape[0]
    bin_edges = np.arange(2.0, 20.5, 0.5)
    hist_stack = np.zeros((length, length, len(bin_edges) - 1), dtype=np.float32)
    counts = np.zeros((length, length), dtype=np.int32)

    for matrix in matrices:
        valid_mask = np.isfinite(matrix) & (matrix >= 0)
        if not np.any(valid_mask):
            continue

        ii, jj = np.where(valid_mask)
        values = matrix[ii, jj]
        bin_ids = np.digitize(values, bin_edges) - 1
        keep_mask = (bin_ids >= 0) & (bin_ids < hist_stack.shape[2])
        if not np.any(keep_mask):
            continue

        ii = ii[keep_mask]
        jj = jj[keep_mask]
        bin_ids = bin_ids[keep_mask]
        np.add.at(hist_stack, (ii, jj, bin_ids), 1)
        np.add.at(counts, (ii, jj), 1)

    mask = counts > 0
    profile = np.zeros_like(hist_stack)
    entropy = np.zeros((length, length, 1), dtype=np.float32)

    valid_i, valid_j = np.where(mask)
    if len(valid_i) > 0:
        probs = hist_stack[valid_i, valid_j] / counts[valid_i, valid_j][:, np.newaxis]
        probs = np.clip(probs, 1e-12, 1.0)
        profile[valid_i, valid_j] = probs
        entropy[valid_i, valid_j, 0] = -np.sum(probs * np.log(probs), axis=1)

    return profile, entropy, mask


def process_single_pdb_task(args):
    group, complex_id, pdb_path, dirs = args

    total_length = get_pdb_full_info(pdb_path)
    if total_length == 0:
        return f"FAIL | {complex_id} (PDB parsing error)"

    matrices = load_all_matrices(dirs["cc"], dirs["q_cc"], group, complex_id, total_length)
    if not matrices:
        return f"SKIP | {complex_id} (No full-length matrices found)"

    profile, entropy, mask = compute_structural_profile(matrices)
    if profile is None:
        return f"SKIP | {complex_id} (Empty matrix data)"

    save_dir = os.path.join(dirs["out"], group, complex_id)
    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, f"{complex_id}.npz")
    np.savez_compressed(out_file, profile=profile, entropy=entropy, mask=mask)
    return f"DONE | {complex_id} (Integrated {len(matrices)} matrices, L={total_length})"


def main():
    root = parse_args()
    dirs = {
        "pdb": os.path.join(root, "pdb"),
        "cc": os.path.join(root, "CCdist"),
        "q_cc": os.path.join(root, "query_CCdist"),
        "out": os.path.join(root, "Profile"),
    }

    if not os.path.exists(dirs["pdb"]):
        print(f"❌ 错误: PDB 目录 {dirs['pdb']} 不存在")
        return

    tasks = []
    for group in os.listdir(dirs["pdb"]):
        group_path = os.path.join(dirs["pdb"], group)
        if not os.path.isdir(group_path):
            continue
        for filename in os.listdir(group_path):
            if filename.endswith(".pdb"):
                complex_id = filename[:-4]
                tasks.append((group, complex_id, os.path.join(group_path, filename), dirs))

    print(f"📂 项目根目录: {root}")
    print(f"🚀 任务总数: {len(tasks)}，正在使用多进程计算...")

    with Pool(processes=min(cpu_count(), 40)) as pool:
        results = pool.map(process_single_pdb_task, tasks)
        done_count = sum(1 for item in results if "DONE" in item)
        fail_count = sum(1 for item in results if "FAIL" in item)
        skip_count = sum(1 for item in results if "SKIP" in item)

        print("\n✨ 计算完成统计:")
        print(f"✅ 成功: {done_count}")
        print(f"⚠️ 跳过: {skip_count}")
        print(f"❌ 失败: {fail_count}")


if __name__ == "__main__":
    main()
