#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from Bio.PDB import NeighborSearch, PDBIO, PDBParser, Select


@dataclass(frozen=True)
class InterfaceTask:
    target: str
    model_name: str
    source_pdb: str
    feature_root: str
    output_pdb_root: str
    output_feature_root: str
    mapping_root: str
    force: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop interface PDBs and canonical features from full-length inputs.")
    parser.add_argument("--pdb-root", required=True, help="Input decoy root directory")
    parser.add_argument("--feature-root", required=True, help="Canonical full-length feature root")
    parser.add_argument("--output-pdb-root", required=True, help="Output root for interface PDBs")
    parser.add_argument("--output-feature-root", required=True, help="Output root for interface features")
    parser.add_argument(
        "--mapping-root",
        default="",
        help="Optional output root for interface residue index mappings. Defaults to <output-pdb-root>/../mapping",
    )
    parser.add_argument("--workers", type=int, default=24, help="Maximum worker processes")
    parser.add_argument("--cb-cutoff", type=float, default=8.0, help="Interface cutoff in angstrom")
    parser.add_argument("--targets", nargs="*", default=None, help="Optional target whitelist")
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing feature bundles instead of failing")
    parser.add_argument("--force", action="store_true", help="Recompute outputs even if they already exist")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_protein_residue(residue) -> bool:
    return residue.id[0] == " "


def get_residue_cb_atom(residue):
    if not is_protein_residue(residue):
        return None
    resname = residue.get_resname().strip().upper()
    if resname == "GLY":
        return residue["CA"] if "CA" in residue else None
    if "CB" in residue:
        return residue["CB"]
    if "CA" in residue:
        return residue["CA"]
    return None


def get_interface_residues_by_cb(structure, cutoff: float) -> set[Tuple[str, object]]:
    model = structure[0]
    rep_atoms = []
    atom_chain = {}
    atom_residue = {}
    valid_chain_ids = set()

    for chain in model.get_chains():
        chain_has_atom = False
        for residue in chain.get_residues():
            atom = get_residue_cb_atom(residue)
            if atom is None:
                continue
            rep_atoms.append(atom)
            atom_chain[atom] = chain.id
            atom_residue[atom] = residue.id
            chain_has_atom = True
        if chain_has_atom:
            valid_chain_ids.add(chain.id)

    if len(valid_chain_ids) < 2:
        return set()

    neighbor_search = NeighborSearch(rep_atoms)
    interface_residues = set()
    for atom in rep_atoms:
        chain_id = atom_chain[atom]
        residue_id = atom_residue[atom]
        for neighbor in neighbor_search.search(atom.coord, cutoff, level="A"):
            other_chain = atom_chain[neighbor]
            if other_chain == chain_id:
                continue
            interface_residues.add((chain_id, residue_id))
            interface_residues.add((other_chain, atom_residue[neighbor]))
    return interface_residues


class InterfaceSelect(Select):
    def __init__(self, interface_residues: set[Tuple[str, object]]):
        super().__init__()
        self.interface_residues = interface_residues

    def accept_residue(self, residue):
        chain_id = residue.get_parent().id
        return (chain_id, residue.id) in self.interface_residues


def build_concat_index_map(structure) -> Dict[Tuple[str, object], int]:
    concat_index = {}
    current = 0
    for chain in structure[0].get_chains():
        for residue in chain.get_residues():
            if not is_protein_residue(residue):
                continue
            concat_index[(chain.id, residue.id)] = current
            current += 1
    return concat_index


def collect_selected_indices(structure, interface_residues: set[Tuple[str, object]]) -> np.ndarray:
    concat_index = build_concat_index_map(structure)
    selected = []
    for chain in structure[0].get_chains():
        for residue in chain.get_residues():
            key = (chain.id, residue.id)
            if key in interface_residues:
                selected.append(concat_index[key])
    return np.asarray(selected, dtype=np.int64)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def save_npz(path: Path, payload: Dict[str, np.ndarray]) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(path, **payload)


def build_old_to_new(selected_indices: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    selected = [int(value) for value in selected_indices.tolist() if int(value) >= 0]
    selected_old = np.asarray(selected, dtype=np.int64)
    old_to_new = {old: new for new, old in enumerate(selected)}
    return selected_old, old_to_new


def slice_square_array(array: np.ndarray, selected: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[np.ix_(selected, selected)]
    if array.ndim == 3:
        if array.shape[0] == array.shape[1]:
            return array[selected][:, selected, :]
        if array.shape[1] == array.shape[2]:
            return array[:, selected][:, :, selected]
    raise ValueError(f"Unsupported square slice shape: {array.shape}")


def crop_voxel(idx: np.ndarray, val: np.ndarray, old_to_new: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    keep_mask = np.asarray([int(item) in old_to_new for item in idx[:, 0]], dtype=bool)
    idx_out = idx[keep_mask].copy()
    val_out = val[keep_mask].copy()
    if idx_out.shape[0] > 0:
        idx_out[:, 0] = np.asarray([old_to_new[int(item)] for item in idx_out[:, 0]])
    return idx_out.astype(np.uint16), val_out.astype(np.float16)


def remap_adj(adj: np.ndarray, old_to_new: Dict[int, int]) -> np.ndarray:
    kept = []
    for left, right in adj:
        if int(left) in old_to_new and int(right) in old_to_new:
            kept.append([old_to_new[int(left)], old_to_new[int(right)]])
    if not kept:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(kept, dtype=np.int64)


def is_decoy_file(path: Path) -> bool:
    if not path.is_file() or path.name.startswith("."):
        return False
    if path.suffix.lower() == ".pdb":
        return True
    if path.suffix:
        return False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for idx, line in enumerate(handle):
                if idx >= 40:
                    break
                if line.startswith(("ATOM", "HETATM", "PFRMAT", "TARGET", "MODEL")):
                    return True
    except OSError:
        return False
    return False


def discover_tasks(
    pdb_root: Path,
    feature_root: Path,
    output_pdb_root: Path,
    output_feature_root: Path,
    mapping_root: Path,
    targets: Iterable[str] | None,
    force: bool,
) -> List[InterfaceTask]:
    allowed = {item.upper() for item in (targets or [])}
    tasks = []
    for target_dir in sorted(pdb_root.iterdir()):
        if not target_dir.is_dir():
            continue
        if allowed and target_dir.name.upper() not in allowed:
            continue
        for decoy_path in sorted(target_dir.iterdir()):
            if not is_decoy_file(decoy_path):
                continue
            tasks.append(
                InterfaceTask(
                    target=target_dir.name,
                    model_name=decoy_path.name,
                    source_pdb=str(decoy_path),
                    feature_root=str(feature_root),
                    output_pdb_root=str(output_pdb_root),
                    output_feature_root=str(output_feature_root),
                    mapping_root=str(mapping_root),
                    force=force,
                )
            )
    return tasks


def crop_base_feature(base_payload: Dict[str, np.ndarray], selected: np.ndarray, old_to_new: Dict[int, int]) -> Dict[str, np.ndarray]:
    output = {}
    if "adj" in base_payload:
        output["adj"] = remap_adj(base_payload["adj"], old_to_new)
    if "idx" in base_payload and "val" in base_payload:
        output["idx"], output["val"] = crop_voxel(base_payload["idx"], base_payload["val"], old_to_new)
    for key in ("phi", "psi"):
        if key in base_payload:
            output[key] = base_payload[key][selected]
    for key in ("omega6d", "theta6d", "phi6d", "maps", "euler"):
        if key in base_payload:
            output[key] = slice_square_array(base_payload[key], selected)
    if "tbt" in base_payload:
        output["tbt"] = base_payload["tbt"][:, selected][:, :, selected]
    for key in ("obt", "prop"):
        if key in base_payload:
            output[key] = base_payload[key][:, selected]
    if "feat" in base_payload:
        feat = base_payload["feat"]
        output["feat"] = feat[:, selected, :] if feat.ndim == 3 else feat[selected]
    return output


def crop_interface_task(task: InterfaceTask, cb_cutoff: float, allow_missing: bool) -> Tuple[bool, str]:
    try:
        source_path = Path(task.source_pdb)
        model_base = source_path.stem if source_path.suffix.lower() == ".pdb" else source_path.name

        output_pdb = Path(task.output_pdb_root) / task.target / task.model_name
        output_mapping = Path(task.mapping_root) / task.target / f"{model_base}.npy"
        output_feature_root = Path(task.output_feature_root)

        base_output = output_feature_root / "base" / task.target / f"{model_base}.npz"
        three_di_output = output_feature_root / "3di" / task.target / f"{model_base}.npz"
        voro_output = output_feature_root / "voro" / task.target / f"{model_base}.npz"
        mpnn_output = output_feature_root / "mpnn" / task.target / f"{model_base}.npz"
        ori_output = output_feature_root / "ori" / task.target / f"{model_base}.npz"
        outputs = [output_pdb, output_mapping, base_output, three_di_output, voro_output, mpnn_output, ori_output]

        if not task.force and all(path.exists() for path in outputs):
            return True, f"[SKIP] {task.target}/{task.model_name}"

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(model_base, str(source_path))
        interface_residues = get_interface_residues_by_cb(structure, cutoff=cb_cutoff)
        if not interface_residues:
            return False, f"[SKIP] No interface residues: {task.target}/{task.model_name}"

        selected_indices = collect_selected_indices(structure, interface_residues)
        selected_old, old_to_new = build_old_to_new(selected_indices)
        if selected_old.size == 0:
            return False, f"[SKIP] Empty interface selection: {task.target}/{task.model_name}"

        ensure_dir(output_pdb.parent)
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(output_pdb), InterfaceSelect(interface_residues))
        ensure_dir(output_mapping.parent)
        np.save(output_mapping, selected_old)

        full_feature_root = Path(task.feature_root)
        base_input = full_feature_root / "base" / task.target / f"{model_base}.npz"
        three_di_input = full_feature_root / "3di" / task.target / f"{model_base}.npz"
        voro_input = full_feature_root / "voro" / task.target / f"{model_base}.npz"
        mpnn_input = full_feature_root / "mpnn" / task.target / f"{model_base}.npz"
        ori_input = full_feature_root / "ori" / f"{task.target}.npz"

        required_inputs = {
            "base": base_input,
            "3di": three_di_input,
            "voro": voro_input,
            "mpnn": mpnn_input,
            "ori": ori_input,
        }
        missing = [name for name, path in required_inputs.items() if not path.exists()]
        if missing:
            message = f"Missing full-length features for {task.target}/{task.model_name}: {', '.join(missing)}"
            if allow_missing:
                return False, f"[SKIP] {message}"
            return False, f"[FAIL] {message}"

        save_npz(base_output, crop_base_feature(load_npz(base_input), selected_old, old_to_new))

        three_di_payload = load_npz(three_di_input)
        save_npz(
            three_di_output,
            {key: value[:, selected_old] if value.ndim == 2 else value for key, value in three_di_payload.items()},
        )

        voro_payload = load_npz(voro_input)
        save_npz(
            voro_output,
            {
                key: (
                    value[selected_old]
                    if value.ndim == 1
                    else slice_square_array(value, selected_old)
                )
                for key, value in voro_payload.items()
            },
        )

        mpnn_payload = load_npz(mpnn_input)
        save_npz(
            mpnn_output,
            {key: value[selected_old] if value.ndim == 2 else value for key, value in mpnn_payload.items()},
        )

        ori_payload = load_npz(ori_input)
        cropped_ori = {}
        for key, value in ori_payload.items():
            if value.ndim == 2 and value.shape[0] == value.shape[1]:
                cropped_ori[key] = slice_square_array(value, selected_old)
            elif value.ndim == 3 and value.shape[0] == value.shape[1]:
                cropped_ori[key] = slice_square_array(value, selected_old)
            else:
                cropped_ori[key] = value
        save_npz(ori_output, cropped_ori)

        return True, f"[OK] {task.target}/{task.model_name} -> L={len(selected_old)}"
    except Exception:
        return False, f"[FAIL] {task.target}/{task.model_name}\n{traceback.format_exc()}"


def main() -> int:
    args = parse_args()
    pdb_root = Path(args.pdb_root).resolve()
    feature_root = Path(args.feature_root).resolve()
    output_pdb_root = Path(args.output_pdb_root).resolve()
    output_feature_root = Path(args.output_feature_root).resolve()
    if args.mapping_root:
        mapping_root = Path(args.mapping_root).resolve()
    else:
        mapping_root = output_pdb_root.parent / "mapping"

    for subdir in ("base", "3di", "voro", "mpnn", "ori"):
        ensure_dir(output_feature_root / subdir)
    ensure_dir(output_pdb_root)
    ensure_dir(mapping_root)

    tasks = discover_tasks(
        pdb_root=pdb_root,
        feature_root=feature_root,
        output_pdb_root=output_pdb_root,
        output_feature_root=output_feature_root,
        mapping_root=mapping_root,
        targets=args.targets,
        force=args.force,
    )
    print(f"[INFO] pdb_root={pdb_root}")
    print(f"[INFO] feature_root={feature_root}")
    print(f"[INFO] output_pdb_root={output_pdb_root}")
    print(f"[INFO] output_feature_root={output_feature_root}")
    print(f"[INFO] tasks={len(tasks)}")

    ok = 0
    skip = 0
    fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(crop_interface_task, task, args.cb_cutoff, args.allow_missing): task
            for task in tasks
        }
        for future in as_completed(futures):
            success, message = future.result()
            print(message, flush=True)
            if success:
                if message.startswith("[SKIP]"):
                    skip += 1
                else:
                    ok += 1
            else:
                if message.startswith("[SKIP]"):
                    skip += 1
                else:
                    fail += 1

    print(f"[SUMMARY] OK={ok} SKIP={skip} FAIL={fail} TOTAL={len(tasks)}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
