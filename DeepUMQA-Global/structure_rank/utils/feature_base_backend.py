#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import sys
import tempfile
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thin CLI wrapper around structure_rank.utils.featurizer.process."
    )
    parser.add_argument("--input_pdb", required=True, help="Input PDB path")
    parser.add_argument("--output_npz", required=True, help="Output NPZ path")
    return parser.parse_args()


def file_has_only_zero_occupancy(path: Path) -> bool:
    atom_lines = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom_lines += 1
        occ_text = line[54:60].strip()
        try:
            occupancy = float(occ_text) if occ_text else 0.0
        except ValueError:
            occupancy = 0.0
        if occupancy > 0:
            return False
    return atom_lines > 0


@contextlib.contextmanager
def normalized_input_pdb(path: Path):
    if not file_has_only_zero_occupancy(path):
        yield path
        return

    with tempfile.TemporaryDirectory(prefix="base_occ_fix_") as tmpdir:
        fixed_path = Path(tmpdir) / path.name
        with path.open("r", encoding="utf-8", errors="ignore") as src, fixed_path.open("w", encoding="utf-8") as dst:
            for raw_line in src:
                line = raw_line.rstrip("\n")
                if line.startswith(("ATOM", "HETATM")):
                    padded = line.ljust(80)
                    line = f"{padded[:54]}{1.00:6.2f}{padded[60:]}"
                dst.write(line + "\n")
        yield fixed_path


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from structure_rank.utils.featurizer import process
    except Exception:
        traceback.print_exc()
        return 2

    try:
        input_pdb = Path(args.input_pdb).resolve()
        with normalized_input_pdb(input_pdb) as effective_pdb:
            ok = process((str(effective_pdb), str(effective_pdb), args.output_npz))
        return 0 if ok else 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
