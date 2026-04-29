import os
import sys
from pathlib import Path

from tqdm import tqdm


def generate_list(input_dir, output_file):
    abs_input_dir = Path(input_dir).resolve()

    if not abs_input_dir.exists():
        print(f"Error: Path {abs_input_dir} does not exist.")
        return

    entries = []
    for target_dir in sorted(abs_input_dir.iterdir()):
        if not target_dir.is_dir() or target_dir.name.startswith('.'):
            continue
        for decoy_path in sorted(target_dir.iterdir()):
            if not decoy_path.is_file():
                continue
            if decoy_path.name.startswith('.'):
                continue
            rel_path = decoy_path.relative_to(abs_input_dir)
            if rel_path.suffix.lower() == '.pdb':
                rel_path = rel_path.with_suffix('')
            entries.append(rel_path.as_posix())

    with open(output_file, 'w', encoding='utf-8') as f:
        for rel_path in tqdm(entries, desc='Writing pdb list'):
            f.write(rel_path + '\n')


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python get_pdblist.py <input_pdb_dir> <output_list_file>")
    else:
        generate_list(sys.argv[1], sys.argv[2])
