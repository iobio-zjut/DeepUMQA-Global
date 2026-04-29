#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import fcntl
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser, is_aa


DEEPUMQA_ROOT = Path(__file__).resolve().parents[3]
PDB_ROOT_DEFAULT = DEEPUMQA_ROOT / "example" / "pdb"
QUERY_ROOT_DEFAULT = DEEPUMQA_ROOT / "example" / "query"
FEATURE_ROOT_DEFAULT = DEEPUMQA_ROOT / "example" / "feature"

SP_PIPELINE = DEEPUMQA_ROOT / "structure_rank" / "utils" / "SP_scripts" / "run_sp_pipeline.py"
BASE_BACKEND = DEEPUMQA_ROOT / "structure_rank" / "utils" / "feature_base_backend.py"
DEFAULT_FOLDSEEK_BIN = os.environ.get("DEEPUMQA_FOLDSEEK_BIN", "").strip()
DEFAULT_MPNN_PYTHON = os.environ.get("DEEPUMQA_MPNN_PYTHON", "").strip()
DEFAULT_VORO_PYTHON = os.environ.get("DEEPUMQA_VORO_PYTHON", "").strip()
DEFAULT_PYROSETTA_PYTHON = os.environ.get("DEEPUMQA_PYROSETTA_PYTHON", "").strip()
DEFAULT_VORO_EXE_DIR = os.environ.get("DEEPUMQA_VORO_EXE_DIR", "").strip()
MPNN_SCRIPT = (
    DEEPUMQA_ROOT / "structure_rank" / "utils" / "GS_LS_SASA" / "mpnn" / "protein_mpnn_run.py"
)
MPNN_COMPRESS = (
    DEEPUMQA_ROOT / "structure_rank" / "utils" / "GS_LS_SASA" / "mpnn" / "compress_npz.py"
)
VORO_AREA_STAGE = (
    DEEPUMQA_ROOT
    / "structure_rank"
    / "utils"
    / "GS_LS_SASA"
    / "voro"
    / "1.area_mutigetfeatureT2.2.py"
)
VORO_NORMAL_STAGE = (
    DEEPUMQA_ROOT
    / "structure_rank"
    / "utils"
    / "GS_LS_SASA"
    / "voro"
    / "1.extract_normalsaf3.py"
)
VORO_AREA_BUILD = (
    DEEPUMQA_ROOT
    / "structure_rank"
    / "utils"
    / "GS_LS_SASA"
    / "voro"
    / "2.area.complex.py"
)
VORO_NORMAL_BUILD = (
    DEEPUMQA_ROOT
    / "structure_rank"
    / "utils"
    / "GS_LS_SASA"
    / "voro"
    / "3.normaltest.py"
)
RUN_PYTHON = sys.executable

BASE_KEYS = {
    "idx",
    "val",
    "phi",
    "psi",
    "omega6d",
    "theta6d",
    "phi6d",
    "feat",
    "adj",
    "tbt",
    "obt",
    "prop",
    "euler",
    "maps",
}
SP_KEYS = {"profile", "entropy", "mask"}
ORI_KEYS = {"profile", "entropy", "mask", "orientation"}
THREEDI_KEYS = {"seq3di"}
VORO_KEYS = {"voro_area", "voro_solvent_area", "normal"}
MPNN_KEYS = {"h_ES_encoder"}
SP_EXECUTABLES = (
    DEEPUMQA_ROOT / "structure_rank" / "utils" / "SP_scripts" / "TMalign",
    DEEPUMQA_ROOT / "structure_rank" / "utils" / "SP_scripts" / "USalign",
)

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
SEQ3DI_ALPHABET = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
SEQ3DI_MAP = {aa: idx for idx, aa in enumerate(SEQ3DI_ALPHABET)}


@dataclass(frozen=True)
class DecoyTask:
    target: str
    decoy: str
    source_path: str
    log_path: str
    feature_root: str
    mpnn_python: Optional[str]
    mpnn_device: str
    foldseek_bin: Optional[str]
    voro_python: str
    voro_exe_dir: Optional[str]
    python_bin: str
    data_root: str
    force: bool


@dataclass(frozen=True)
class TargetTask:
    target: str
    query_path: Optional[str]
    log_path: str
    feature_root: str
    data_root: str
    force: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified local feature extraction pipeline for DeepUMQA-G.")
    parser.add_argument("--pdb-root", default=str(PDB_ROOT_DEFAULT), help="Decoy root directory")
    parser.add_argument("--query-root", default=str(QUERY_ROOT_DEFAULT), help="AF3 query root directory")
    parser.add_argument("--feature-root", default=str(FEATURE_ROOT_DEFAULT), help="Feature output root directory")
    parser.add_argument("--workers", type=int, default=24, help="Maximum worker processes")
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Optional target whitelist. Case-sensitive against pdb directory names.",
    )
    parser.add_argument(
        "--decoy-only",
        action="store_true",
        help="Skip target SP/ORI feature generation and only process decoy features.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if the output NPZ already exists")
    parser.add_argument(
        "--mpnn-device",
        choices=("auto", "cpu", "gpu"),
        default=os.environ.get("MPNN_DEVICE", "auto"),
        help="Device flag forwarded to ProteinMPNN",
    )
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for local helper scripts")
    parser.add_argument(
        "--foldseek-bin",
        default=DEFAULT_FOLDSEEK_BIN,
        help="Foldseek executable path; falls back to PATH when omitted",
    )
    parser.add_argument(
        "--mpnn-python",
        default=DEFAULT_MPNN_PYTHON,
        help="ProteinMPNN Python executable; falls back to current Python when omitted",
    )
    parser.add_argument(
        "--voro-python",
        default=DEFAULT_VORO_PYTHON or sys.executable,
        help="Python executable for Voronota helper scripts",
    )
    parser.add_argument(
        "--pyrosetta-python",
        default=DEFAULT_PYROSETTA_PYTHON or sys.executable,
        help="Python executable used when probing PyRosetta availability",
    )
    parser.add_argument(
        "--voro-exe-dir",
        default=DEFAULT_VORO_EXE_DIR,
        help="Voronota executable directory",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(log_path: Path, message: str) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(f"{now_str()} | {message}\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_traceback(log_path: Path, feature_name: str, target: str, decoy: Optional[str], exc: BaseException) -> None:
    header = f"TRACEBACK | feature={feature_name} | target={target}"
    if decoy:
        header += f" | decoy={decoy}"
    append_log(log_path, header)
    append_log(log_path, traceback.format_exc().rstrip())
    append_log(log_path, f"ERROR | {type(exc).__name__}: {exc}")


def truncate_text(text: str, limit: int = 2400) -> str:
    compact = " ".join((text or "").splitlines())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]} ...[truncated]"


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def run_base_backend_local(
    *,
    staged_pdb: Path,
    tmp_path: Path,
    python_bin: str,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess[str]:
    return run_command(
        [
            python_bin,
            str(BASE_BACKEND),
            "--input_pdb",
            str(staged_pdb),
            "--output_npz",
            str(tmp_path),
        ],
        cwd=DEEPUMQA_ROOT,
        timeout=timeout,
    )


def run_command_with_files(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path],
    input_path: Path,
    output_path: Path,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess[str]:
    with input_path.open("r", encoding="utf-8", errors="ignore") as handle_in:
        with output_path.open("w", encoding="utf-8") as handle_out:
            return subprocess.run(
                list(cmd),
                cwd=str(cwd) if cwd else None,
                stdin=handle_in,
                stdout=handle_out,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )


def candidate_query_names(target: str) -> List[str]:
    upper = target.upper()
    candidates = [upper]
    if upper.endswith("O"):
        candidates.append(upper[:-1])
    return list(dict.fromkeys(candidates))


def candidate_target_names(name: str) -> List[str]:
    upper = name.upper()
    candidates = [upper]
    if upper.endswith("O"):
        candidates.append(upper[:-1])
    return list(dict.fromkeys(candidates))


def build_query_index(query_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if not query_root.exists():
        return index
    for path in sorted(query_root.iterdir()):
        if path.is_file() and path.suffix.lower() == ".pdb":
            index[path.stem.upper()] = path
    return index


def resolve_query_path(target: str, query_index: Dict[str, Path]) -> Optional[Path]:
    for candidate in candidate_query_names(target):
        if candidate in query_index:
            return query_index[candidate]
    return None


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


def discover_decoys(target_dir: Path) -> List[Path]:
    return sorted(path for path in target_dir.iterdir() if is_decoy_file(path))


def discover_targets(
    pdb_root: Path,
    query_root: Path,
    feature_root: Path,
    selected_targets: Optional[Iterable[str]],
    *,
    skip_target_tasks: bool,
    mpnn_python: Optional[str],
    mpnn_device: str,
    foldseek_bin: Optional[str],
    voro_python: str,
    voro_exe_dir: Optional[str],
    python_bin: str,
    force: bool,
) -> Tuple[List[TargetTask], List[DecoyTask], Dict[str, int]]:
    allow = set(selected_targets or [])
    allow_upper = {name.upper() for name in allow}
    target_tasks: List[TargetTask] = []
    decoy_tasks: List[DecoyTask] = []
    counts = {"targets": 0, "decoys": 0}
    query_index = build_query_index(query_root) if not skip_target_tasks else {}
    seen_targets: set[str] = set()

    for target_dir in sorted(pdb_root.iterdir()):
        if not target_dir.is_dir():
            continue
        target = target_dir.name
        if allow and not set(candidate_target_names(target)).intersection(allow_upper):
            continue
        seen_targets.add(target.upper())

        log_path = feature_root / "logs" / f"{target}.log"
        data_root = str(feature_root.parent)
        if not skip_target_tasks:
            query_path = resolve_query_path(target, query_index)
            target_tasks.append(
                TargetTask(
                    target=target,
                    query_path=str(query_path) if query_path else None,
                    log_path=str(log_path),
                    feature_root=str(feature_root),
                    data_root=data_root,
                    force=force,
                )
            )

        decoys = discover_decoys(target_dir)
        counts["targets"] += 1
        counts["decoys"] += len(decoys)

        for decoy_path in decoys:
            decoy_name = decoy_path.stem if decoy_path.suffix.lower() == ".pdb" else decoy_path.name
            decoy_tasks.append(
                DecoyTask(
                    target=target,
                    decoy=decoy_name,
                    source_path=str(decoy_path),
                    log_path=str(log_path),
                    feature_root=str(feature_root),
                    mpnn_python=mpnn_python,
                    mpnn_device=mpnn_device,
                    foldseek_bin=foldseek_bin,
                    voro_python=voro_python,
                    voro_exe_dir=voro_exe_dir,
                    python_bin=python_bin,
                    data_root=data_root,
                    force=force,
                )
            )

    if not skip_target_tasks:
        for query_name in sorted(query_index):
            if allow and query_name.upper() not in allow_upper:
                continue
            if query_name.upper() in seen_targets:
                continue
            log_path = feature_root / "logs" / f"{query_name}.log"
            target_tasks.append(
                TargetTask(
                    target=query_name,
                    query_path=str(query_index[query_name]),
                    log_path=str(log_path),
                    feature_root=str(feature_root),
                    data_root=str(feature_root.parent),
                    force=force,
                )
            )
            counts["targets"] += 1

    return target_tasks, decoy_tasks, counts


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def validate_npz_keys(path: Path, required_keys: Sequence[str]) -> Tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "file missing or empty"
    try:
        with np.load(path, allow_pickle=True) as data:
            missing = [key for key in required_keys if key not in data.files]
        if missing:
            return False, f"missing keys: {', '.join(missing)}"
        return True, "ok"
    except Exception as exc:
        return False, f"failed to read npz: {exc}"


def validate_base_npz(path: Path, expected_length: Optional[int] = None) -> Tuple[bool, str]:
    ok, reason = validate_npz_keys(path, BASE_KEYS)
    if not ok:
        return ok, reason
    try:
        with np.load(path, allow_pickle=True) as data:
            feat = data["feat"]
            obt = data["obt"]
            prop = data["prop"]
            tbt = data["tbt"]
            euler = data["euler"]
            maps = data["maps"]
        if feat.ndim != 3:
            return False, f"unexpected feat shape {feat.shape}"
        length = feat.shape[1]
        if expected_length is not None and length != expected_length:
            return False, f"expected length {expected_length}, got {length}"
        if obt.ndim != 2 or obt.shape[1] != length:
            return False, f"unexpected obt shape {obt.shape}"
        if prop.ndim != 2 or prop.shape[1] != length:
            return False, f"unexpected prop shape {prop.shape}"
        if tbt.ndim != 3 or tbt.shape[1] != length or tbt.shape[2] != length:
            return False, f"unexpected tbt shape {tbt.shape}"
        if euler.ndim != 3 or euler.shape[0] != length or euler.shape[1] != length:
            return False, f"unexpected euler shape {euler.shape}"
        if maps.ndim != 3 or maps.shape[0] != length or maps.shape[1] != length:
            return False, f"unexpected maps shape {maps.shape}"
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def validate_ori_npz(path: Path, expected_length: Optional[int] = None) -> Tuple[bool, str]:
    ok, reason = validate_npz_keys(path, ORI_KEYS)
    if not ok:
        return ok, reason
    try:
        with np.load(path, allow_pickle=True) as data:
            profile = data["profile"]
            entropy = data["entropy"]
            mask = data["mask"]
            orientation = data["orientation"]
        if profile.ndim != 3 or profile.shape[0] != profile.shape[1]:
            return False, f"unexpected profile shape {profile.shape}"
        length = profile.shape[0]
        if expected_length is not None and length != expected_length:
            return False, f"expected length {expected_length}, got {length}"
        if entropy.ndim < 2 or entropy.shape[0] != length or entropy.shape[1] != length:
            return False, f"unexpected entropy shape {entropy.shape}"
        if mask.ndim != 2 or mask.shape[0] != length or mask.shape[1] != length:
            return False, f"unexpected mask shape {mask.shape}"
        if orientation.ndim != 3 or orientation.shape[0] != length or orientation.shape[1] != length:
            return False, f"unexpected orientation shape {orientation.shape}"
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def validate_3di_npz(path: Path, expected_length: Optional[int] = None) -> Tuple[bool, str]:
    ok, reason = validate_npz_keys(path, THREEDI_KEYS)
    if not ok:
        return ok, reason
    try:
        with np.load(path, allow_pickle=True) as data:
            seq3di = data["seq3di"]
        if seq3di.ndim != 2 or seq3di.shape[0] != 20:
            return False, f"unexpected seq3di shape {seq3di.shape}"
        if expected_length is not None and seq3di.shape[1] != expected_length:
            return False, f"expected length {expected_length}, got {seq3di.shape[1]}"
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def validate_voro_npz(path: Path, expected_length: Optional[int] = None) -> Tuple[bool, str]:
    ok, reason = validate_npz_keys(path, VORO_KEYS)
    if not ok:
        return ok, reason
    try:
        with np.load(path, allow_pickle=True) as data:
            voro_area = data["voro_area"]
            voro_solvent_area = data["voro_solvent_area"]
            normal = data["normal"]
        if voro_area.ndim != 2 or voro_area.shape[0] != voro_area.shape[1]:
            return False, f"unexpected voro_area shape {voro_area.shape}"
        if normal.ndim != 3 or normal.shape[-1] != 3:
            return False, f"unexpected normal shape {normal.shape}"
        if expected_length is not None:
            if voro_area.shape[0] != expected_length:
                return False, f"expected area length {expected_length}, got {voro_area.shape[0]}"
            if voro_solvent_area.shape[0] != expected_length:
                return False, f"expected solvent length {expected_length}, got {voro_solvent_area.shape[0]}"
            if normal.shape[0] != expected_length or normal.shape[1] != expected_length:
                return False, f"expected normal length {expected_length}, got {normal.shape}"
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def validate_mpnn_npz(path: Path, expected_length: Optional[int] = None) -> Tuple[bool, str]:
    ok, reason = validate_npz_keys(path, MPNN_KEYS)
    if not ok:
        return ok, reason
    try:
        with np.load(path, allow_pickle=True) as data:
            hidden = data["h_ES_encoder"]
        if hidden.ndim != 2:
            return False, f"unexpected h_ES_encoder shape {hidden.shape}"
        if expected_length is not None and hidden.shape[0] != expected_length:
            return False, f"expected length {expected_length}, got {hidden.shape[0]}"
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def get_base_length(path: Path) -> int:
    with np.load(path, allow_pickle=True) as data:
        return int(data["feat"].shape[1])


def get_3di_length(path: Path) -> int:
    with np.load(path, allow_pickle=True) as data:
        return int(data["seq3di"].shape[1])


def get_voro_length(path: Path) -> int:
    with np.load(path, allow_pickle=True) as data:
        return int(data["voro_solvent_area"].shape[0])


def get_mpnn_length(path: Path) -> int:
    with np.load(path, allow_pickle=True) as data:
        return int(data["h_ES_encoder"].shape[0])


def get_ori_length(path: Path) -> int:
    with np.load(path, allow_pickle=True) as data:
        return int(data["profile"].shape[0])


def validate_decoy_feature_bundle(
    *,
    base_path: Path,
    three_di_path: Path,
    voro_path: Path,
    mpnn_path: Path,
    ori_path: Path,
) -> Tuple[bool, str]:
    validators = {
        "base": (validate_base_npz, base_path),
        "3di": (validate_3di_npz, three_di_path),
        "voro": (validate_voro_npz, voro_path),
        "mpnn": (validate_mpnn_npz, mpnn_path),
        "ori": (validate_ori_npz, ori_path),
    }
    length_readers = {
        "base": get_base_length,
        "3di": get_3di_length,
        "voro": get_voro_length,
        "mpnn": get_mpnn_length,
        "ori": get_ori_length,
    }
    lengths = {}

    for feature_name, (validator, path) in validators.items():
        ok, reason = validator(path)
        if not ok:
            return False, f"{feature_name}: {reason}"
        lengths[feature_name] = length_readers[feature_name](path)

    unique_lengths = sorted(set(lengths.values()))
    if len(unique_lengths) != 1:
        detail = ", ".join(f"{name}={length}" for name, length in sorted(lengths.items()))
        return False, f"length mismatch | {detail}"
    return True, f"length={unique_lengths[0]}"


def atomic_replace(src_path: Path, dst_path: Path) -> None:
    ensure_dir(dst_path.parent)
    shutil.move(str(src_path), str(dst_path))


def save_npz_atomically(final_path: Path, payload: Dict[str, np.ndarray], temp_dir: Path) -> None:
    ensure_dir(final_path.parent)
    tmp_path = temp_dir / f"{final_path.name}.tmp.npz"
    np.savez_compressed(tmp_path, **payload)
    atomic_replace(tmp_path, final_path)


def _rewrite_residue_id(line: str, new_resseq: int, insertion_code: str = " ") -> str:
    padded = line.rstrip("\n").ljust(80)
    return f"{padded[:22]}{new_resseq:4d}{insertion_code[:1]}{padded[27:]}"


def stage_as_pdb(source_path: Path, staged_path: Path) -> None:
    ensure_dir(staged_path.parent)
    residue_numbers: Dict[Tuple[str, str, str, str], int] = {}
    chain_counters: Dict[str, int] = defaultdict(int)
    last_residue_meta: Dict[str, Tuple[str, int]] = {}

    with source_path.open("r", encoding="utf-8", errors="ignore") as src, staged_path.open("w", encoding="utf-8") as dst:
        for raw_line in src:
            line = raw_line.rstrip("\n")
            record = line[:6]
            if record in ("ATOM  ", "HETATM", "ANISOU"):
                chain_id = line[21:22]
                resseq = line[22:26]
                icode = line[26:27]
                resname = line[17:20]
                residue_key = (chain_id, resseq, icode, resname)
                new_resseq = residue_numbers.get(residue_key)
                if new_resseq is None:
                    chain_counters[chain_id] += 1
                    new_resseq = chain_counters[chain_id]
                    residue_numbers[residue_key] = new_resseq
                rewritten = _rewrite_residue_id(line, new_resseq)
                if record in ("ATOM  ", "HETATM"):
                    last_residue_meta[chain_id] = (resname, new_resseq)
                dst.write(rewritten + "\n")
                continue

            if record == "TER   ":
                chain_id = line[21:22]
                last_meta = last_residue_meta.get(chain_id)
                if last_meta is not None:
                    resname, new_resseq = last_meta
                    padded = line.ljust(80)
                    line = f"{padded[:17]}{resname:>3}{padded[20:21]}{chain_id}{new_resseq:4d} {padded[27:]}"
                dst.write(line + "\n")
                continue

            dst.write(line + "\n")


def count_residues(pdb_path: Path) -> int:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    first_model = next(structure.get_models())
    count = 0
    for chain in first_model:
        for residue in chain:
            if residue.id[0] == " " and is_aa(residue, standard=False):
                count += 1
    return count


def get_chain_names(pdb_path: Path) -> List[str]:
    chain_names = set()
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
                chain_id = line[21].strip()
                if chain_id:
                    chain_names.add(chain_id)
    return sorted(chain_names)


def one_hot_encode_seq3di(sequence: str) -> np.ndarray:
    one_hot = np.zeros((20, len(sequence)), dtype=np.float32)
    for idx, aa in enumerate(sequence):
        aa_idx = SEQ3DI_MAP.get(aa)
        if aa_idx is not None:
            one_hot[aa_idx, idx] = 1.0
    return one_hot


def build_concat_sequence_for_one_pdb(pdb_path: Path, fasta_sequences: Dict[str, object]) -> str:
    model_name = pdb_path.stem
    pdb_file = pdb_path.name
    chain_names = get_chain_names(pdb_path)
    all_keys = list(fasta_sequences.keys())
    concat: List[str] = []

    for idx, chain in enumerate(chain_names):
        candidates = [f"{model_name}_{chain}", f"{pdb_file}_{chain}", f"{model_name}{chain}", f"{pdb_file} {idx}"]
        found_seq = None
        for candidate in candidates:
            if candidate in fasta_sequences:
                found_seq = str(fasta_sequences[candidate].seq)
                break
        if found_seq is None:
            for fasta_id in all_keys:
                if model_name in fasta_id and (chain in fasta_id or str(idx) in fasta_id):
                    found_seq = str(fasta_sequences[fasta_id].seq)
                    break
        if found_seq:
            concat.append(found_seq)
    if concat:
        return "".join(concat)
    if len(fasta_sequences) == 1:
        only_record = next(iter(fasta_sequences.values()))
        return str(only_record.seq)
    return ""


def ensure_sp_executables() -> None:
    for path in SP_EXECUTABLES:
        if path.exists() and not os.access(path, os.X_OK):
            path.chmod(path.stat().st_mode | 0o111)


def resize_square(array: np.ndarray, expected_length: int, dtype: np.dtype) -> np.ndarray:
    out = np.zeros((expected_length, expected_length), dtype=dtype)
    rows = min(expected_length, array.shape[0])
    cols = min(expected_length, array.shape[1])
    out[:rows, :cols] = array[:rows, :cols]
    return out


def resize_vector(array: np.ndarray, expected_length: int, dtype: np.dtype) -> np.ndarray:
    out = np.zeros(expected_length, dtype=dtype)
    size = min(expected_length, array.shape[0])
    out[:size] = array[:size]
    return out


def resize_normal(array: np.ndarray, expected_length: int) -> np.ndarray:
    out = np.zeros((expected_length, expected_length, 3), dtype=np.float32)
    rows = min(expected_length, array.shape[0])
    cols = min(expected_length, array.shape[1])
    out[:rows, :cols, :] = array[:rows, :cols, :]
    return out


def summarize_result(result: subprocess.CompletedProcess[str]) -> str:
    parts = []
    if result.stdout:
        parts.append(f"stdout={truncate_text(result.stdout)}")
    if result.stderr:
        parts.append(f"stderr={truncate_text(result.stderr)}")
    if not parts:
        return f"return_code={result.returncode}"
    return f"return_code={result.returncode} | " + " | ".join(parts)


def resolve_mpnn_python(explicit_path: str) -> Optional[str]:
    for raw in [explicit_path, DEFAULT_MPNN_PYTHON, sys.executable]:
        if raw and Path(raw).exists():
            return str(Path(raw).resolve())
    return sys.executable if Path(sys.executable).exists() else None


def resolve_foldseek_bin(explicit_path: str) -> Optional[str]:
    for raw in [explicit_path, DEFAULT_FOLDSEEK_BIN]:
        if raw and Path(raw).exists():
            return str(Path(raw).resolve())
    return shutil.which("foldseek")


def resolve_voro_exe_dir(explicit_path: str) -> Optional[Path]:
    for raw in [explicit_path, DEFAULT_VORO_EXE_DIR]:
        if raw and Path(raw).exists():
            return Path(raw).resolve()
    return None


def resolve_mpnn_device(requested: str) -> str:
    requested = (requested or "auto").lower()
    if requested == "cpu":
        return "cpu"
    if requested == "gpu":
        return "gpu"
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False
    return "gpu" if has_cuda else "cpu"


def result_record(kind: str, name: str, feature_statuses: Dict[str, str], feature_messages: Dict[str, str], ready: bool) -> Dict[str, object]:
    return {
        "kind": kind,
        "name": name,
        "feature_statuses": feature_statuses,
        "feature_messages": feature_messages,
        "ready": ready,
    }


def maybe_skip_base(final_path: Path, *, force: bool) -> Tuple[bool, str]:
    if force:
        return False, "force enabled"
    ok, reason = validate_base_npz(final_path)
    return ok, reason


def build_base_fallback_payload(staged_pdb: Path) -> Dict[str, np.ndarray]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(staged_pdb.stem, str(staged_pdb))
    first_model = next(structure.get_models())

    residues = [
        residue
        for chain in first_model
        for residue in chain
        if residue.id[0] == " " and is_aa(residue, standard=False)
    ]
    residue_count = len(residues)

    payload: Dict[str, np.ndarray] = {
        "idx": np.zeros((0, 5), dtype=np.uint16),
        "val": np.zeros((0,), dtype=np.float16),
        "phi": np.zeros(residue_count, dtype=np.float16),
        "psi": np.zeros(residue_count, dtype=np.float16),
        "omega6d": np.zeros((residue_count, residue_count), dtype=np.float16),
        "theta6d": np.zeros((residue_count, residue_count), dtype=np.float16),
        "phi6d": np.zeros((residue_count, residue_count), dtype=np.float16),
        "feat": np.zeros((1, residue_count, 7), dtype=np.float16),
        "adj": np.zeros((0, 2), dtype=np.int32),
        "tbt": np.zeros((10, residue_count, residue_count), dtype=np.float16),
        "obt": np.zeros((13, residue_count), dtype=np.float16),
        "prop": np.zeros((55, residue_count), dtype=np.float16),
        "euler": np.zeros((residue_count, residue_count, 6), dtype=np.float16),
        "maps": np.zeros((residue_count, residue_count, 4), dtype=np.float16),
    }
    return payload


def extract_base_feature(
    *,
    staged_pdb: Path,
    final_path: Path,
    temp_root: Path,
    python_bin: str,
    force: bool,
) -> Tuple[str, str]:
    ok, reason = maybe_skip_base(final_path, force=force)
    if ok:
        return "skipped", reason

    tmp_path = temp_root / "base_feature.npz"
    result = run_base_backend_local(
        staged_pdb=staged_pdb,
        tmp_path=tmp_path,
        python_bin=python_bin,
        timeout=7200,
    )
    if result.returncode != 0:
        raise RuntimeError(f"base backend failed locally: {summarize_result(result)}")

    ok, reason = validate_npz_keys(tmp_path, BASE_KEYS)
    if not ok:
        raise RuntimeError(f"invalid base npz: {reason}")

    atomic_replace(tmp_path, final_path)
    return "success", "generated"


def extract_3di_feature(
    *,
    staged_pdb: Path,
    final_path: Path,
    temp_root: Path,
    residue_count: int,
    foldseek_bin: Optional[str],
    force: bool,
) -> Tuple[str, str]:
    if not force:
        ok, reason = validate_3di_npz(final_path, residue_count)
        if ok:
            return "skipped", reason

    if not foldseek_bin:
        save_npz_atomically(
            final_path,
            {"seq3di": np.zeros((20, residue_count), dtype=np.float32)},
            temp_root,
        )
        return "fallback", "foldseek binary unavailable; wrote zeros"

    db_prefix = temp_root / "seq3di_db"
    fasta_path = temp_root / f"{staged_pdb.stem}_3di.fasta"
    commands = [
        [foldseek_bin, "createdb", str(staged_pdb), str(db_prefix), "--threads", "1"],
        [foldseek_bin, "lndb", f"{db_prefix}_h", f"{db_prefix}_ss_h"],
        [foldseek_bin, "convert2fasta", f"{db_prefix}_ss", str(fasta_path)],
    ]

    try:
        for cmd in commands:
            result = run_command(cmd, timeout=3600)
            if result.returncode != 0:
                raise RuntimeError(summarize_result(result))

        fasta_sequences = {record.id: record for record in SeqIO.parse(str(fasta_path), "fasta")}
        concat_seq = build_concat_sequence_for_one_pdb(staged_pdb, fasta_sequences)
        if not concat_seq:
            raise RuntimeError("empty 3di sequence")
        if len(concat_seq) != residue_count:
            raise RuntimeError(f"3di residue mismatch: expected {residue_count}, got {len(concat_seq)}")

        save_npz_atomically(final_path, {"seq3di": one_hot_encode_seq3di(concat_seq)}, temp_root)
        return "success", "generated"
    except Exception as exc:
        save_npz_atomically(
            final_path,
            {"seq3di": np.zeros((20, residue_count), dtype=np.float32)},
            temp_root,
        )
        return "fallback", f"{exc}; wrote zeros"


def extract_voro_feature(
    *,
    staged_root: Path,
    target: str,
    decoy: str,
    final_path: Path,
    temp_root: Path,
    residue_count: int,
    voro_python: str,
    voro_exe_dir: Optional[str],
    force: bool,
) -> Tuple[str, str]:
    if not force:
        ok, reason = validate_voro_npz(final_path, residue_count)
        if ok:
            return "skipped", reason

    resolved_voro_exe_dir = Path(voro_exe_dir).resolve() if voro_exe_dir else None
    if resolved_voro_exe_dir is None or not resolved_voro_exe_dir.exists():
        save_npz_atomically(
            final_path,
            {
                "voro_area": np.zeros((residue_count, residue_count), dtype=np.float32),
                "voro_solvent_area": np.zeros(residue_count, dtype=np.float32),
                "normal": np.zeros((residue_count, residue_count, 3), dtype=np.float32),
            },
            temp_root,
        )
        return "fallback", "voronota executable directory missing; wrote zeros"

    target_list = temp_root / "targets.txt"
    target_list.write_text(f"{target}\n", encoding="utf-8")
    area_txt_root = temp_root / "voro_area_txt"
    normal_txt_root = temp_root / "voro_normal_txt"
    area_npz_root = temp_root / "voro_area_npz"
    normal_npz_root = temp_root / "voro_normal_npz"

    try:
        area_target_dir = ensure_dir(area_txt_root / target)
        normal_target_dir = ensure_dir(normal_txt_root / target)
        area_balls = temp_root / f"{decoy}.area.balls.txt"
        normal_balls = temp_root / f"{decoy}.normal.balls.txt"
        area_contact = area_target_dir / f"{decoy}.txt"
        normal_contact = normal_target_dir / f"{decoy}.txt"

        result = run_command_with_files(
            ["./voronota", "get-balls-from-atoms-file", "--annotated"],
            cwd=resolved_voro_exe_dir,
            input_path=staged_root / target / f"{decoy}.pdb",
            output_path=area_balls,
            timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(summarize_result(result))

        result = run_command_with_files(
            ["./voronota", "calculate-contacts", "--annotated"],
            cwd=resolved_voro_exe_dir,
            input_path=area_balls,
            output_path=area_contact,
            timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(summarize_result(result))

        result = run_command_with_files(
            ["./voronota-normal", "get-balls-from-atoms-file", "--annotated"],
            cwd=resolved_voro_exe_dir,
            input_path=staged_root / target / f"{decoy}.pdb",
            output_path=normal_balls,
            timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(summarize_result(result))

        result = run_command_with_files(
            ["./voronota-normal", "calculate-contacts", "--annotated"],
            cwd=resolved_voro_exe_dir,
            input_path=normal_balls,
            output_path=normal_contact,
            timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(summarize_result(result))

        for cmd in (
            [voro_python, str(VORO_AREA_BUILD), str(target_list), str(area_txt_root), str(area_npz_root)],
            [voro_python, str(VORO_NORMAL_BUILD), str(target_list), str(normal_txt_root), str(normal_npz_root)],
        ):
            result = run_command(cmd, cwd=resolved_voro_exe_dir, timeout=7200)
            if result.returncode != 0:
                raise RuntimeError(summarize_result(result))

        area_npz = area_npz_root / target / f"{decoy}.npz"
        normal_npz = normal_npz_root / target / f"{decoy}.npz"
        if not area_npz.exists():
            raise FileNotFoundError(f"missing area npz: {area_npz}")
        if not normal_npz.exists():
            raise FileNotFoundError(f"missing normal npz: {normal_npz}")

        area_data = load_npz(area_npz)
        normal_data = load_npz(normal_npz)
        voro_area = resize_square(area_data["voro_area"].astype(np.float32), residue_count, np.float32)
        voro_solvent_area = resize_vector(
            area_data["voro_solvent_area"].astype(np.float32), residue_count, np.float32
        )
        normal = resize_normal(normal_data["normal"].astype(np.float32), residue_count)
        save_npz_atomically(
            final_path,
            {
                "voro_area": voro_area,
                "voro_solvent_area": voro_solvent_area,
                "normal": normal,
            },
            temp_root,
        )
        return "success", "generated"
    except Exception as exc:
        save_npz_atomically(
            final_path,
            {
                "voro_area": np.zeros((residue_count, residue_count), dtype=np.float32),
                "voro_solvent_area": np.zeros(residue_count, dtype=np.float32),
                "normal": np.zeros((residue_count, residue_count, 3), dtype=np.float32),
            },
            temp_root,
        )
        return "fallback", f"{exc}; wrote zeros"


def extract_mpnn_feature(
    *,
    staged_pdb: Path,
    target: str,
    decoy: str,
    final_path: Path,
    temp_root: Path,
    residue_count: int,
    mpnn_python: Optional[str],
    mpnn_device: str,
    force: bool,
) -> Tuple[str, str]:
    if not force:
        ok, reason = validate_mpnn_npz(final_path, residue_count)
        if ok:
            return "skipped", reason

    if not mpnn_python:
        raise RuntimeError("ProteinMPNN python interpreter unavailable.")
    if not MPNN_SCRIPT.exists():
        raise FileNotFoundError(f"missing ProteinMPNN script: {MPNN_SCRIPT}")

    def build_cmd(device_name: str, out_root: Path) -> list:
        return [
            mpnn_python,
            str(MPNN_SCRIPT),
            "--pdb_path",
            str(staged_pdb),
            "--num_seq_per_target",
            "1",
            "--sampling_temp",
            "0.1",
            "--seed",
            "3",
            "--device",
            device_name,
            "--out_folder",
            str(out_root),
            "--output_path",
            str(out_root),
        ]

    requested = (mpnn_device or "auto").lower()
    primary = resolve_mpnn_device(requested)
    device_plan = [primary]
    if primary == "gpu":
        device_plan.append("cpu")

    errors = []
    for device_name in device_plan:
        out_root = temp_root / f"mpnn_output_{device_name}"
        result = run_command(build_cmd(device_name, out_root), cwd=MPNN_SCRIPT.parent, timeout=7200)
        if result.returncode != 0:
            errors.append(f"device={device_name}: {summarize_result(result)}")
            continue

        raw_npz = out_root / target / f"{decoy}_h_ES_encoder.npz"
        if not raw_npz.exists():
            errors.append(f"device={device_name}: missing raw mpnn npz: {raw_npz}")
            continue

        compress = run_command([mpnn_python, str(MPNN_COMPRESS), str(raw_npz)], cwd=MPNN_SCRIPT.parent, timeout=1800)
        if compress.returncode != 0:
            errors.append(f"device={device_name}: compress failed: {summarize_result(compress)}")
            continue

        ok, reason = validate_mpnn_npz(raw_npz, residue_count)
        if not ok:
            errors.append(f"device={device_name}: invalid mpnn npz: {reason}")
            continue

        atomic_replace(raw_npz, final_path)
        suffix = "" if device_name == primary else f" after fallback from {primary}"
        return "success", f"generated on {device_name}{suffix}"

    raise RuntimeError("; ".join(errors) if errors else "mpnn feature generation failed")


def process_decoy_task(task: DecoyTask) -> Dict[str, object]:
    target = task.target
    decoy = task.decoy
    log_path = Path(task.log_path)
    feature_root = Path(task.feature_root)
    staged_root: Optional[Path] = None

    append_log(log_path, f"DECOY | {decoy} | START | source={task.source_path}")
    feature_statuses: Dict[str, str] = {}
    feature_messages: Dict[str, str] = {}

    try:
        with tempfile.TemporaryDirectory(prefix=f".feature_{target}_{decoy}_", dir=task.data_root) as tmp_dir:
            temp_root = Path(tmp_dir)
            staged_root = temp_root / "pdb"
            staged_pdb = staged_root / target / f"{decoy}.pdb"
            stage_as_pdb(Path(task.source_path), staged_pdb)
            residue_count = count_residues(staged_pdb)
            if residue_count <= 0:
                raise RuntimeError("parsed residue count is zero")

            base_path = feature_root / "base" / target / f"{decoy}.npz"
            three_di_path = feature_root / "3di" / target / f"{decoy}.npz"
            voro_path = feature_root / "voro" / target / f"{decoy}.npz"
            mpnn_path = feature_root / "mpnn" / target / f"{decoy}.npz"
            ori_path = feature_root / "ori" / f"{target}.npz"

            feature_plan = [
                ("base", lambda: extract_base_feature(
                    staged_pdb=staged_pdb,
                    final_path=base_path,
                    temp_root=temp_root,
                    python_bin=task.python_bin,
                    force=task.force,
                )),
                ("3di", lambda: extract_3di_feature(
                    staged_pdb=staged_pdb,
                    final_path=three_di_path,
                    temp_root=temp_root,
                    residue_count=residue_count,
                    foldseek_bin=task.foldseek_bin,
                    force=task.force,
                )),
                ("voro", lambda: extract_voro_feature(
                    staged_root=staged_root,
                    target=target,
                    decoy=decoy,
                    final_path=voro_path,
                    temp_root=temp_root,
                    residue_count=residue_count,
                    voro_python=task.voro_python,
                    voro_exe_dir=task.voro_exe_dir,
                    force=task.force,
                )),
                ("mpnn", lambda: extract_mpnn_feature(
                    staged_pdb=staged_pdb,
                    target=target,
                    decoy=decoy,
                    final_path=mpnn_path,
                    temp_root=temp_root,
                    residue_count=residue_count,
                    mpnn_python=task.mpnn_python,
                    mpnn_device=task.mpnn_device,
                    force=task.force,
                )),
            ]

            for feature_name, runner in feature_plan:
                try:
                    status, message = runner()
                    feature_statuses[feature_name] = status
                    feature_messages[feature_name] = message
                    append_log(log_path, f"DECOY | {decoy} | {feature_name} | {status.upper()} | {message}")
                except Exception as exc:
                    feature_statuses[feature_name] = "failed"
                    feature_messages[feature_name] = str(exc)
                    append_log(log_path, f"DECOY | {decoy} | {feature_name} | FAILED | {exc}")
                    append_traceback(log_path, feature_name, target, decoy, exc)

            bundle_ok, bundle_reason = validate_decoy_feature_bundle(
                base_path=base_path,
                three_di_path=three_di_path,
                voro_path=voro_path,
                mpnn_path=mpnn_path,
                ori_path=ori_path,
            )
            feature_statuses["consistency"] = "success" if bundle_ok else "failed"
            feature_messages["consistency"] = bundle_reason
            append_log(
                log_path,
                f"DECOY | {decoy} | consistency | {feature_statuses['consistency'].upper()} | {bundle_reason}",
            )

            ready = all(feature_statuses.get(name) != "failed" for name in ["base", "3di", "voro", "mpnn", "consistency"])
            append_log(log_path, f"DECOY | {decoy} | END | ready={ready}")
            return result_record(
                kind="decoy",
                name=f"{target}/{decoy}",
                feature_statuses=feature_statuses,
                feature_messages=feature_messages,
                ready=ready,
            )
    except Exception as exc:
        append_log(log_path, f"DECOY | {decoy} | FAILED | {exc}")
        append_traceback(log_path, "decoy_setup", target, decoy, exc)
        for feature_name in ("base", "3di", "voro", "mpnn", "consistency"):
            feature_statuses.setdefault(feature_name, "failed")
            feature_messages.setdefault(feature_name, str(exc))
        return result_record(
            kind="decoy",
            name=f"{target}/{decoy}",
            feature_statuses=feature_statuses,
            feature_messages=feature_messages,
            ready=False,
        )


def process_target_task(task: TargetTask) -> Dict[str, object]:
    target = task.target
    log_path = Path(task.log_path)
    feature_root = Path(task.feature_root)
    sp_path = feature_root / "sp" / f"{target}.npz"
    ori_path = feature_root / "ori" / f"{target}.npz"
    feature_statuses: Dict[str, str] = {}
    feature_messages: Dict[str, str] = {}

    append_log(log_path, f"TARGET | {target} | SP_ORI | START | query={task.query_path or 'MISSING'} | mode=local")

    if not task.force:
        sp_ok, sp_reason = validate_npz_keys(sp_path, SP_KEYS)
        ori_ok, ori_reason = validate_ori_npz(ori_path)
        if sp_ok and ori_ok:
            feature_statuses["sp"] = "skipped"
            feature_messages["sp"] = sp_reason
            feature_statuses["ori"] = "skipped"
            feature_messages["ori"] = ori_reason
            append_log(log_path, f"TARGET | {target} | SP_ORI | SKIPPED | outputs already exist")
            return result_record(
                kind="target",
                name=target,
                feature_statuses=feature_statuses,
                feature_messages=feature_messages,
                ready=True,
            )

    if not task.query_path:
        message = "query structure not found"
        feature_statuses["sp"] = "failed"
        feature_messages["sp"] = message
        feature_statuses["ori"] = "failed"
        feature_messages["ori"] = message
        append_log(log_path, f"TARGET | {target} | SP_ORI | FAILED | {message}")
        return result_record(
            kind="target",
            name=target,
            feature_statuses=feature_statuses,
            feature_messages=feature_messages,
            ready=False,
        )

    try:
        ensure_sp_executables()
        with tempfile.TemporaryDirectory(prefix=f".spori_{target}_", dir=task.data_root) as tmp_dir:
            temp_root = Path(tmp_dir)
            input_dir = temp_root / "input"
            staged_query = input_dir / f"{target}.pdb"
            stage_as_pdb(Path(task.query_path), staged_query)
            output_dir = temp_root / "feature"

            sp_args = [
                RUN_PYTHON,
                str(SP_PIPELINE),
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(output_dir),
                "--samples",
                target,
            ]
            result = run_command(
                sp_args,
                cwd=DEEPUMQA_ROOT,
                timeout=14400,
            )
            if result.returncode != 0:
                raise RuntimeError(summarize_result(result))

            tmp_sp = output_dir / "sp" / f"{target}.npz"
            tmp_ori = output_dir / "ori" / f"{target}.npz"
            sp_ok, sp_reason = validate_npz_keys(tmp_sp, SP_KEYS)
            ori_ok, ori_reason = validate_ori_npz(tmp_ori)
            if not sp_ok:
                raise RuntimeError(f"invalid sp npz: {sp_reason}")
            if not ori_ok:
                raise RuntimeError(f"invalid ori npz: {ori_reason}")

            atomic_replace(tmp_sp, sp_path)
            atomic_replace(tmp_ori, ori_path)
            feature_statuses["sp"] = "success"
            feature_messages["sp"] = "generated"
            feature_statuses["ori"] = "success"
            feature_messages["ori"] = "generated"
            append_log(log_path, f"TARGET | {target} | sp | SUCCESS | generated")
            append_log(log_path, f"TARGET | {target} | ori | SUCCESS | generated")
            return result_record(
                kind="target",
                name=target,
                feature_statuses=feature_statuses,
                feature_messages=feature_messages,
                ready=True,
            )
    except Exception as exc:
        feature_statuses["sp"] = "failed"
        feature_messages["sp"] = str(exc)
        feature_statuses["ori"] = "failed"
        feature_messages["ori"] = str(exc)
        append_log(log_path, f"TARGET | {target} | SP_ORI | FAILED | {exc}")
        append_traceback(log_path, "sp_ori", target, None, exc)
        return result_record(
            kind="target",
            name=target,
            feature_statuses=feature_statuses,
            feature_messages=feature_messages,
            ready=False,
        )


def write_summary(
    *,
    feature_root: Path,
    discovery_counts: Dict[str, int],
    results: List[Dict[str, object]],
) -> Tuple[str, int]:
    decoy_results = [item for item in results if item["kind"] == "decoy"]
    target_results = [item for item in results if item["kind"] == "target"]
    feature_counters: Dict[str, Counter] = defaultdict(Counter)
    failed_items: List[str] = []

    for item in results:
        statuses = item["feature_statuses"]
        messages = item["feature_messages"]
        for feature_name, status in statuses.items():
            feature_counters[feature_name][status] += 1
            if status == "failed":
                failed_items.append(f"{item['kind']} {item['name']} | {feature_name} | {messages.get(feature_name, '')}")

    decoy_success = sum(1 for item in decoy_results if item["ready"])
    decoy_failed = len(decoy_results) - decoy_success
    target_success = sum(1 for item in target_results if item["ready"])
    target_failed = len(target_results) - target_success

    lines = [
        f"Summary generated at {now_str()}",
        f"Targets discovered: {discovery_counts['targets']}",
        f"Decoys discovered: {discovery_counts['decoys']}",
        f"Decoy success count: {decoy_success}",
        f"Decoy failed count: {decoy_failed}",
        f"Target sp/ori success count: {target_success}",
        f"Target sp/ori failed count: {target_failed}",
        "",
        "Per-feature completion:",
    ]

    for feature_name in sorted(feature_counters):
        counter = feature_counters[feature_name]
        parts = [f"{key}={counter[key]}" for key in sorted(counter)]
        lines.append(f"{feature_name}: " + ", ".join(parts))

    lines.append("")
    lines.append("Failed items:")
    if failed_items:
        lines.extend(failed_items)
    else:
        lines.append("None")

    summary_text = "\n".join(lines)
    summary_path = feature_root / "logs" / "pipeline_summary.log"
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    return summary_text, decoy_failed + target_failed


def prepare_output_dirs(feature_root: Path) -> None:
    for subdir in ("base", "3di", "voro", "mpnn", "sp", "ori", "logs"):
        ensure_dir(feature_root / subdir)


def main() -> int:
    args = parse_args()
    global RUN_PYTHON
    RUN_PYTHON = args.python_bin
    pdb_root = Path(args.pdb_root).resolve()
    query_root = Path(args.query_root).resolve()
    feature_root = Path(args.feature_root).resolve()
    prepare_output_dirs(feature_root)

    mpnn_python = resolve_mpnn_python(args.mpnn_python)
    foldseek_bin = resolve_foldseek_bin(args.foldseek_bin)
    voro_python = str(Path(args.voro_python).resolve()) if args.voro_python else sys.executable
    voro_exe_dir = resolve_voro_exe_dir(args.voro_exe_dir)

    target_tasks, decoy_tasks, discovery_counts = discover_targets(
        pdb_root,
        query_root,
        feature_root,
        args.targets,
        skip_target_tasks=args.decoy_only,
        mpnn_python=mpnn_python,
        mpnn_device=resolve_mpnn_device(args.mpnn_device),
        foldseek_bin=foldseek_bin,
        voro_python=voro_python,
        voro_exe_dir=str(voro_exe_dir) if voro_exe_dir else "",
        python_bin=args.python_bin,
        force=args.force,
    )

    header = [
        f"START | feature_root={feature_root}",
        f"DISCOVERY | targets={discovery_counts['targets']} decoys={discovery_counts['decoys']}",
        "BACKEND | mode=local",
        f"BACKEND | python_bin={args.python_bin}",
        f"BACKEND | mpnn_python={mpnn_python or 'MISSING'}",
        f"BACKEND | mpnn_device={resolve_mpnn_device(args.mpnn_device)} (requested={args.mpnn_device})",
        f"BACKEND | foldseek_bin={foldseek_bin or 'MISSING'}",
        f"BACKEND | voro_python={voro_python}",
        f"BACKEND | voro_exe_dir={voro_exe_dir or 'MISSING'}",
        f"BACKEND | pyrosetta_python={args.pyrosetta_python or 'MISSING'}",
    ]
    ensure_dir(feature_root / "logs")
    (feature_root / "logs" / "pipeline_runtime.log").write_text("\n".join(header) + "\n", encoding="utf-8")

    results: List[Dict[str, object]] = []

    for task in target_tasks:
        label = f"target:{task.target}"
        try:
            results.append(process_target_task(task))
        except Exception as exc:
            results.append(
                result_record(
                    kind="internal",
                    name=label,
                    feature_statuses={"pipeline": "failed"},
                    feature_messages={"pipeline": str(exc)},
                    ready=False,
                )
            )

    max_workers = max(1, min(args.workers, len(decoy_tasks) or 1))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for task in decoy_tasks:
            future = executor.submit(process_decoy_task, task)
            future_map[future] = f"decoy:{task.target}/{task.decoy}"

        for future in as_completed(future_map):
            label = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    result_record(
                        kind="internal",
                        name=label,
                        feature_statuses={"pipeline": "failed"},
                        feature_messages={"pipeline": str(exc)},
                        ready=False,
                    )
                )

    summary_text, failure_count = write_summary(
        feature_root=feature_root,
        discovery_counts=discovery_counts,
        results=results,
    )
    print(summary_text)
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
