import argparse
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, is_aa


def get_total_length(pdb_path: Path) -> int:
    structure = PDBParser(QUIET=True).get_structure("query", str(pdb_path))
    return sum(
        1
        for chain in structure[0]
        for res in chain
        if res.id[0] == " " and is_aa(res, standard=False)
    )


def iter_matrix_paths(root: Path, sample: str):
    template_dir = root / "CCdist" / sample / sample
    if template_dir.is_dir():
        for path in sorted(template_dir.glob("*.npy")):
            yield path

    for path in [
        root / "query_CCdist" / sample / f"{sample}.npy",
        root / "query_CCdist" / f"{sample}.npy",
    ]:
        if path.is_file():
            yield path
            break


def build_profile(root: Path, sample: str):
    pdb_path = root / "pdb" / sample / f"{sample}.pdb"
    out_dir = root / "Profile" / sample / sample
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample}.npz"

    total_length = get_total_length(pdb_path)
    bin_edges = np.arange(2.0, 20.5, 0.5, dtype=np.float32)
    hist_stack = np.zeros((total_length, total_length, len(bin_edges) - 1), dtype=np.float32)
    counts = np.zeros((total_length, total_length), dtype=np.int32)

    used = 0
    for matrix_path in iter_matrix_paths(root, sample):
        matrix = np.load(matrix_path, mmap_mode="r")
        if matrix.ndim != 2 or matrix.shape != (total_length, total_length):
            continue

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
        used += 1

    mask = counts > 0
    entropy = np.zeros((total_length, total_length, 1), dtype=np.float32)

    for i in range(total_length):
        row_mask = mask[i]
        if not np.any(row_mask):
            continue
        row_counts = counts[i, row_mask].astype(np.float32, copy=False)
        hist_stack[i, row_mask, :] /= row_counts[:, None]
        probs = np.clip(hist_stack[i, row_mask, :], 1e-12, 1.0)
        entropy[i, row_mask, 0] = -np.sum(probs * np.log(probs), axis=1)

    np.savez_compressed(out_path, profile=hist_stack, entropy=entropy, mask=mask)
    return out_path, used, total_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--sample", required=True)
    args = parser.parse_args()

    out_path, used, total_length = build_profile(Path(args.root), args.sample)
    print(f"DONE {out_path} matrices={used} L={total_length}")


if __name__ == "__main__":
    main()
