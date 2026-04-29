#!/usr/bin/env python3
"""Unified wrapper for the existing SP extraction pipelines.

This script keeps the legacy monomer/complex SP workflows intact, but normalizes
their outputs into:
1. feature_root/sp/*.npz
2. feature_root/sp/temp/<run_id>/...
3. feature_root/ori/*.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, is_aa


SCRIPT_DIR = Path(__file__).resolve().parent
COMPLEX_PIPELINE_SH = SCRIPT_DIR / "SP.sh"
MONOMER_PIPELINE_SH = SCRIPT_DIR / "SP_monomer.sh"
ORI_PIPELINE_PY = SCRIPT_DIR / "run_ori.py"
TMALIGN_BIN = SCRIPT_DIR / "TMalign"
USALIGN_BIN = SCRIPT_DIR / "USalign"
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parents[4] if len(SCRIPT_DIR.parents) >= 5 else Path.cwd()

PYTORCH_PYTHON = Path(os.environ.get("DEEPUMQA_PYTHON_BIN", sys.executable))
STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}


@dataclass
class ChainStat:
    chain_id: str
    aa_residues: int
    all_residues: int
    het_residues: int


@dataclass
class DetectionInfo:
    parser: str
    protein_chain_count: int
    atom_chain_count: int
    chain_stats: List[ChainStat] = field(default_factory=list)
    fallback_used: bool = False
    note: str = ""


@dataclass
class SampleRecord:
    name: str
    source_path: str
    source_suffix: str
    detected_type: str
    selected_type: str
    detection: DetectionInfo
    staged_input: Optional[str] = None
    branch_work_root: Optional[str] = None
    final_result: Optional[str] = None
    orientation_result: Optional[str] = None
    branch_log: Optional[str] = None
    branch_return_code: Optional[int] = None
    status: str = "pending"
    reason: str = ""
    audit: Dict[str, str] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified entry point for the existing SP extraction workflows."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "pdb",
        help="Directory containing input sample folders or structure files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "feature",
        help="Feature root directory. Outputs will be written into output_dir/sp and output_dir/ori.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "monomer", "complex"),
        default="auto",
        help="Force a branch for all samples, or auto-detect per sample.",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional sample names to process. By default all samples are scanned.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit after discovery and filtering.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only perform discovery, classification, staging plan, and preflight checks.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_logging(output_dir: Path, run_id: str) -> Tuple[logging.Logger, Path]:
    log_dir = ensure_dir(output_dir / "temp" / run_id / "logs")
    log_path = log_dir / f"{run_id}.log"

    logger = logging.getLogger("run_sp_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger, log_path


def run_command(
    cmd: Sequence[str],
    *,
    logger: logging.Logger,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    logger.info("RUN | cwd=%s | %s", cwd or Path.cwd(), " ".join(cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="[dry-run]\n", stderr="")

    result = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    if log_path is not None:
        ensure_dir(log_path.parent)
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"COMMAND: {' '.join(cmd)}\n")
            handle.write(f"RETURN_CODE: {result.returncode}\n")
            handle.write("\n[STDOUT]\n")
            handle.write(result.stdout)
            handle.write("\n[STDERR]\n")
            handle.write(result.stderr)

    if result.stdout.strip():
        logger.info("STDOUT | %s", truncate_log_block(result.stdout))
    if result.stderr.strip():
        logger.warning("STDERR | %s", truncate_log_block(result.stderr))
    logger.info("RET  | %s", result.returncode)
    return result


def truncate_log_block(text: str, limit: int = 1500) -> str:
    clean = " ".join(text.strip().splitlines())
    if len(clean) <= limit:
        return clean
    return f"{clean[:limit]} ...[truncated]"


def select_structure_file(sample_name: str, entry: Path) -> Path:
    if entry.is_file() and entry.suffix.lower() in STRUCTURE_SUFFIXES:
        return entry

    if not entry.is_dir():
        raise ValueError(f"{entry} is neither a supported structure file nor a sample directory")

    candidates = sorted(
        path
        for path in entry.rglob("*")
        if path.is_file() and path.suffix.lower() in STRUCTURE_SUFFIXES
    )
    if not candidates:
        raise ValueError(f"no structure file found under {entry}")

    exact_name = [path for path in candidates if path.stem == sample_name]
    if len(exact_name) == 1:
        return exact_name[0]

    pdb_candidates = [path for path in candidates if path.suffix.lower() == ".pdb"]
    if len(pdb_candidates) == 1:
        return pdb_candidates[0]

    if len(candidates) == 1:
        return candidates[0]

    candidate_str = ", ".join(str(path.name) for path in candidates[:10])
    raise ValueError(f"multiple candidate structure files found: {candidate_str}")


def discover_samples(
    input_dir: Path,
    *,
    selected_names: Optional[Iterable[str]],
    max_samples: Optional[int],
) -> List[Tuple[str, Path]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

    selected_set = set(selected_names or [])
    discovered: List[Tuple[str, Path]] = []

    for entry in sorted(input_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        sample_name = entry.stem if entry.is_file() else entry.name
        if selected_set and sample_name not in selected_set:
            continue
        structure_file = select_structure_file(sample_name, entry)
        discovered.append((sample_name, structure_file))
        if max_samples is not None and len(discovered) >= max_samples:
            break

    if selected_set:
        found = {name for name, _ in discovered}
        missing = sorted(selected_set - found)
        if missing:
            raise FileNotFoundError(f"requested samples not found: {', '.join(missing)}")

    return discovered


def parse_structure_chain_stats(structure_path: Path) -> DetectionInfo:
    suffix = structure_path.suffix.lower()
    parser_name = "MMCIFParser" if suffix in {".cif", ".mmcif"} else "PDBParser"
    parser = MMCIFParser(QUIET=True) if parser_name == "MMCIFParser" else PDBParser(QUIET=True)

    structure = parser.get_structure(structure_path.stem, str(structure_path))
    first_model = next(structure.get_models())

    chain_stats: List[ChainStat] = []
    protein_chain_count = 0

    for chain in first_model:
        aa_residues = 0
        all_residues = 0
        het_residues = 0
        for residue in chain.get_residues():
            if residue.id[0] == " ":
                all_residues += 1
                if is_aa(residue, standard=False):
                    aa_residues += 1
            else:
                het_residues += 1
        if aa_residues > 0:
            protein_chain_count += 1
        chain_stats.append(
            ChainStat(
                chain_id=chain.id.strip() or "_",
                aa_residues=aa_residues,
                all_residues=all_residues,
                het_residues=het_residues,
            )
        )

    atom_chain_count = sum(
        1 for chain_stat in chain_stats if chain_stat.all_residues > 0 or chain_stat.het_residues > 0
    )
    return DetectionInfo(
        parser=parser_name,
        protein_chain_count=protein_chain_count,
        atom_chain_count=atom_chain_count,
        chain_stats=chain_stats,
        fallback_used=False,
        note="Protein chain count derived from amino-acid residues in the first model.",
    )


def fallback_detect_from_text(structure_path: Path) -> DetectionInfo:
    chain_ids = []
    with structure_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 22:
                chain_id = line[21].strip() or "_"
                chain_ids.append(chain_id)
    unique_chain_ids = sorted(set(chain_ids))
    return DetectionInfo(
        parser="text_fallback",
        protein_chain_count=len(unique_chain_ids),
        atom_chain_count=len(unique_chain_ids),
        chain_stats=[
            ChainStat(chain_id=chain_id, aa_residues=0, all_residues=0, het_residues=0)
            for chain_id in unique_chain_ids
        ],
        fallback_used=True,
        note="Biopython parsing failed; fell back to distinct ATOM/HETATM chain identifiers.",
    )


def detect_structure_type(structure_path: Path) -> Tuple[str, DetectionInfo]:
    try:
        detection = parse_structure_chain_stats(structure_path)
    except Exception as exc:
        detection = fallback_detect_from_text(structure_path)
        detection.note = f"{detection.note} Original parser error: {exc}"

    protein_chain_count = detection.protein_chain_count
    if protein_chain_count <= 0:
        # When only the text fallback is available, treat one chain as monomer and
        # more than one chain as complex. This remains conservative for legacy flows.
        structure_type = "monomer" if detection.atom_chain_count <= 1 else "complex"
    else:
        structure_type = "monomer" if protein_chain_count == 1 else "complex"
    return structure_type, detection


def convert_or_copy_to_pdb(source_path: Path, dest_pdb_path: Path) -> None:
    ensure_dir(dest_pdb_path.parent)
    suffix = source_path.suffix.lower()
    if suffix == ".pdb":
        shutil.copy2(source_path, dest_pdb_path)
        return

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(dest_pdb_path.stem, str(source_path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(dest_pdb_path))


def stage_samples(
    samples: List[SampleRecord],
    branch_root: Path,
    *,
    branch_name: str,
    logger: logging.Logger,
    dry_run: bool,
) -> None:
    for sample in samples:
        staged_input = branch_root / "pdb" / sample.name / f"{sample.name}.pdb"
        sample.staged_input = str(staged_input)
        sample.branch_work_root = str(branch_root)
        if dry_run:
            logger.info("DRY | stage %s -> %s", sample.source_path, staged_input)
            continue
        convert_or_copy_to_pdb(Path(sample.source_path), staged_input)
        logger.info("STAGE | %s sample=%s -> %s", branch_name, sample.name, staged_input)


def build_script_env() -> Dict[str, str]:
    env = os.environ.copy()
    if PYTORCH_PYTHON.exists():
        env["PATH"] = f"{PYTORCH_PYTHON.parent}:{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def local_python() -> str:
    return str(PYTORCH_PYTHON) if PYTORCH_PYTHON.exists() else sys.executable


def preflight_checks(
    samples: List[SampleRecord],
    *,
    logger: logging.Logger,
    dry_run: bool,
) -> None:
    if not samples:
        raise RuntimeError("no valid samples discovered")

    for path in (COMPLEX_PIPELINE_SH, MONOMER_PIPELINE_SH, ORI_PIPELINE_PY, TMALIGN_BIN, USALIGN_BIN):
        if not path.exists():
            raise FileNotFoundError(f"required file missing: {path}")
    for path in (TMALIGN_BIN, USALIGN_BIN):
        if not os.access(path, os.X_OK):
            raise PermissionError(f"required executable is not executable: {path}")

    hostname = socket.gethostname().split(".")[0]
    logger.info("HOST | current host: %s", hostname)
    logger.info("ENV  | wrapper python: %s", sys.executable)
    logger.info("ENV  | local pipeline python: %s", local_python())

    preflight_dir = ensure_dir(Path(samples[0].branch_work_root or Path.cwd()) / "_preflight_logs")

    commands = [
        (
            "local_foldseek_env",
            [
                local_python(),
                "-c",
                "import shutil, sys; path = shutil.which('foldseek'); print(path or 'MISSING'); raise SystemExit(0 if path else 1)",
            ],
        ),
        (
            "local_downpdb_python",
            [
                local_python(),
                "-c",
                "import requests; from Bio import PDB; import sys; print(sys.executable)",
            ],
        ),
        (
            "local_orientation_python",
            [
                local_python(),
                "-c",
                "import numpy, scipy, tqdm; from Bio import PDB; import sys; print(sys.executable)",
            ],
        ),
    ]

    for label, cmd in commands:
        log_path = preflight_dir / f"{label}.log"
        result = run_command(
            cmd,
            logger=logger,
            cwd=SCRIPT_DIR,
            env=build_script_env(),
            log_path=log_path,
            dry_run=dry_run,
        )
        if result.returncode != 0:
            raise RuntimeError(f"preflight check failed: {label}. See {log_path}")


def run_branch_pipeline(
    branch_name: str,
    pipeline_script: Path,
    branch_root: Path,
    *,
    logger: logging.Logger,
    temp_run_dir: Path,
    dry_run: bool,
) -> subprocess.CompletedProcess[str]:
    run_label = temp_run_dir.name
    log_path = ensure_dir(temp_run_dir / "logs") / f"{run_label}_{branch_name}_pipeline.log"
    return run_command(
        ["bash", str(pipeline_script), str(branch_root)],
        logger=logger,
        cwd=SCRIPT_DIR,
        env=build_script_env(),
        log_path=log_path,
        dry_run=dry_run,
    )


def complex_expected_outputs(branch_root: Path, sample_name: str) -> Dict[str, Path]:
    return {
        "foldseek_m8": branch_root / "search_result" / "result" / sample_name / f"{sample_name}_results.m8",
        "foldseek_report": branch_root / "search_result" / "result" / sample_name / f"{sample_name}_results.m8_report",
        "templates_dir": branch_root / "templates",
        "query_cat_dir": branch_root / "query_cat" / sample_name / sample_name,
        "template_cat_dir": branch_root / "templates_cat" / sample_name / sample_name,
        "align_dir": branch_root / "align_results" / sample_name / sample_name,
        "query_ccdist_dir": branch_root / "query_CCdist" / sample_name,
        "template_ccdist_dir": branch_root / "CCdist" / sample_name / sample_name,
        "profile_npz": branch_root / "Profile" / sample_name / sample_name / f"{sample_name}.npz",
    }


def monomer_expected_outputs(branch_root: Path, sample_name: str) -> Dict[str, Path]:
    return {
        "foldseek_m8": branch_root / "search_result" / "result" / sample_name / f"{sample_name}_results.m8",
        "templates_cat_dir": branch_root / "templates_cat" / sample_name / sample_name,
        "align_file": branch_root / "align_results" / sample_name / f"{sample_name}_alignment.txt",
        "query_ccdist": branch_root / "CCdist" / sample_name / f"{sample_name}_q.npy",
        "template_ccdist_dir": branch_root / "CCdist" / sample_name,
        "profile_npz": branch_root / "Profile" / "profile_npz" / f"{sample_name}.npz",
    }


def path_has_payload(path: Path) -> bool:
    if path.is_file():
        return path.exists() and path.stat().st_size > 0
    if path.is_dir():
        return any(path.iterdir())
    return False


def audit_sample_outputs(sample: SampleRecord, branch_root: Path) -> Tuple[Optional[Path], Dict[str, str], str]:
    expectations = (
        monomer_expected_outputs(branch_root, sample.name)
        if sample.selected_type == "monomer"
        else complex_expected_outputs(branch_root, sample.name)
    )

    audit: Dict[str, str] = {}
    for label, path in expectations.items():
        audit[label] = "present" if path_has_payload(path) else f"missing:{path}"

    profile_npz = expectations["profile_npz"]
    if profile_npz.exists():
        return profile_npz, audit, "success"

    ordered_labels = list(expectations.keys())
    for label in ordered_labels:
        if audit[label].startswith("missing:"):
            return None, audit, f"missing output at step '{label}'"
    return None, audit, "final profile npz missing"


def save_sample_record(temp_run_dir: Path, sample: SampleRecord) -> None:
    record_dir = ensure_dir(temp_run_dir / "records")
    record_path = record_dir / f"{sample.name}.json"
    with record_path.open("w", encoding="utf-8") as handle:
        json.dump(sample_to_json(sample), handle, ensure_ascii=False, indent=2)


def sample_to_json(sample: SampleRecord) -> Dict[str, object]:
    payload = asdict(sample)
    payload["detection"]["chain_stats"] = [
        asdict(chain_stat) for chain_stat in sample.detection.chain_stats
    ]
    return payload


def copy_final_result(sample: SampleRecord, results_dir: Path, result_npz: Path) -> Path:
    ensure_dir(results_dir)
    destination = results_dir / f"{sample.name}.npz"
    shutil.copy2(result_npz, destination)
    return destination


def write_orientation_manifest(temp_run_dir: Path, samples: List[SampleRecord], ori_dir: Path) -> Path:
    manifest_path = temp_run_dir / "orientation_tasks.tsv"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            if not sample.staged_input or not sample.final_result:
                continue
            output_npz = ori_dir / f"{sample.name}.npz"
            handle.write(
                "\t".join(
                    [
                        sample.name,
                        sample.staged_input,
                        sample.final_result,
                        str(output_npz),
                    ]
                )
                + "\n"
            )
    return manifest_path


def run_orientation_pipeline(
    manifest_path: Path,
    *,
    logger: logging.Logger,
    temp_run_dir: Path,
    dry_run: bool,
) -> subprocess.CompletedProcess[str]:
    log_path = ensure_dir(temp_run_dir / "logs") / f"{temp_run_dir.name}_orientation_pipeline.log"
    return run_command(
        [local_python(), str(ORI_PIPELINE_PY), "--manifest", str(manifest_path)],
        logger=logger,
        cwd=SCRIPT_DIR,
        env=build_script_env(),
        log_path=log_path,
        dry_run=dry_run,
    )


def orientation_result_ready(npz_path: Path) -> bool:
    if not npz_path.exists() or npz_path.stat().st_size == 0:
        return False
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            return "orientation" in data.files
    except Exception:
        return False


def write_summary(temp_run_dir: Path, samples: List[SampleRecord]) -> Path:
    summary_path = temp_run_dir / "summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "sample",
                "detected_type",
                "selected_type",
                "status",
                "reason",
                "source_path",
                "staged_input",
                "final_result",
                "orientation_result",
                "branch_return_code",
                "protein_chain_count",
                "atom_chain_count",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.name,
                    sample.detected_type,
                    sample.selected_type,
                    sample.status,
                    sample.reason,
                    sample.source_path,
                    sample.staged_input or "",
                    sample.final_result or "",
                    sample.orientation_result or "",
                    sample.branch_return_code if sample.branch_return_code is not None else "",
                    sample.detection.protein_chain_count,
                    sample.detection.atom_chain_count,
                ]
            )
    return summary_path


def prepare_samples(
    discovered: List[Tuple[str, Path]],
    *,
    mode: str,
) -> List[SampleRecord]:
    samples: List[SampleRecord] = []
    for sample_name, structure_file in discovered:
        detected_type, detection = detect_structure_type(structure_file)
        selected_type = detected_type if mode == "auto" else mode
        samples.append(
            SampleRecord(
                name=sample_name,
                source_path=str(structure_file),
                source_suffix=structure_file.suffix.lower(),
                detected_type=detected_type,
                selected_type=selected_type,
                detection=detection,
            )
        )
    return samples


def split_by_branch(samples: List[SampleRecord]) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    monomer = [sample for sample in samples if sample.selected_type == "monomer"]
    complex_samples = [sample for sample in samples if sample.selected_type == "complex"]
    return monomer, complex_samples


def main() -> int:
    args = parse_args()
    feature_root = args.output_dir.resolve()
    sp_output_dir = ensure_dir(feature_root / "sp")
    ori_output_dir = ensure_dir(feature_root / "ori")
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger, main_log_path = configure_logging(sp_output_dir, run_id)

    logger.info("START | run_id=%s", run_id)
    logger.info("ARGS  | %s", vars(args))
    logger.info("NOTE  | SP_main.sh is intentionally not used because it hardcodes an outdated script path.")
    logger.info("OUT   | sp=%s ori=%s", sp_output_dir, ori_output_dir)

    try:
        discovered = discover_samples(
            args.input_dir.resolve(),
            selected_names=args.samples,
            max_samples=args.max_samples,
        )
        logger.info("SCAN  | discovered %d samples", len(discovered))
        for sample_name, structure_file in discovered:
            logger.info("FOUND | %s -> %s", sample_name, structure_file)

        samples = prepare_samples(discovered, mode=args.mode)
        for sample in samples:
            logger.info(
                "TYPE  | %s detected=%s selected=%s protein_chains=%d atom_chains=%d note=%s",
                sample.name,
                sample.detected_type,
                sample.selected_type,
                sample.detection.protein_chain_count,
                sample.detection.atom_chain_count,
                sample.detection.note,
            )

        monomer_samples, complex_samples = split_by_branch(samples)

        temp_run_dir = ensure_dir(sp_output_dir / "temp" / run_id)
        results_dir = sp_output_dir
        monomer_root = temp_run_dir / "monomer"
        complex_root = temp_run_dir / "complex"

        if monomer_samples:
            stage_samples(
                monomer_samples,
                monomer_root,
                branch_name="monomer",
                logger=logger,
                dry_run=args.dry_run,
            )
        if complex_samples:
            stage_samples(
                complex_samples,
                complex_root,
                branch_name="complex",
                logger=logger,
                dry_run=args.dry_run,
            )

        for sample in samples:
            save_sample_record(temp_run_dir, sample)

        preflight_checks(samples, logger=logger, dry_run=args.dry_run)

        branch_results: Dict[str, subprocess.CompletedProcess[str]] = {}
        if complex_samples:
            branch_results["complex"] = run_branch_pipeline(
                "complex",
                COMPLEX_PIPELINE_SH,
                complex_root,
                logger=logger,
                temp_run_dir=temp_run_dir,
                dry_run=args.dry_run,
            )
        if monomer_samples:
            branch_results["monomer"] = run_branch_pipeline(
                "monomer",
                MONOMER_PIPELINE_SH,
                monomer_root,
                logger=logger,
                temp_run_dir=temp_run_dir,
                dry_run=args.dry_run,
            )

        for sample in complex_samples:
            sample.branch_return_code = branch_results["complex"].returncode
            sample.branch_log = str(temp_run_dir / "logs" / f"{temp_run_dir.name}_complex_pipeline.log")
            if args.dry_run:
                sample.status = "dry_run"
                sample.reason = "discovery, staging plan, and preflight only"
            else:
                result_npz, audit, reason = audit_sample_outputs(sample, complex_root)
                sample.audit = audit
                if result_npz is not None:
                    destination = copy_final_result(sample, results_dir, result_npz)
                    sample.final_result = str(destination)
                    sample.status = "sp_ready"
                    sample.reason = "legacy complex pipeline completed"
                else:
                    sample.status = "failed"
                    sample.reason = reason
            save_sample_record(temp_run_dir, sample)

        for sample in monomer_samples:
            sample.branch_return_code = branch_results["monomer"].returncode
            sample.branch_log = str(temp_run_dir / "logs" / f"{temp_run_dir.name}_monomer_pipeline.log")
            if args.dry_run:
                sample.status = "dry_run"
                sample.reason = "discovery, staging plan, and preflight only"
            else:
                result_npz, audit, reason = audit_sample_outputs(sample, monomer_root)
                sample.audit = audit
                if result_npz is not None:
                    destination = copy_final_result(sample, results_dir, result_npz)
                    sample.final_result = str(destination)
                    sample.status = "sp_ready"
                    sample.reason = "legacy monomer pipeline completed"
                else:
                    sample.status = "failed"
                    sample.reason = reason
            save_sample_record(temp_run_dir, sample)

        if not args.dry_run:
            orientation_inputs = [sample for sample in samples if sample.status == "sp_ready"]
            if orientation_inputs:
                manifest_path = write_orientation_manifest(temp_run_dir, orientation_inputs, ori_output_dir)
                orientation_run = run_orientation_pipeline(
                    manifest_path,
                    logger=logger,
                    temp_run_dir=temp_run_dir,
                    dry_run=args.dry_run,
                )
                orientation_log = temp_run_dir / "logs" / f"{temp_run_dir.name}_orientation_pipeline.log"
                for sample in orientation_inputs:
                    sample.audit["orientation_run_rc"] = str(orientation_run.returncode)
                    sample.audit["orientation_log"] = str(orientation_log)
                    orientation_npz = ori_output_dir / f"{sample.name}.npz"
                    if orientation_result_ready(orientation_npz):
                        sample.orientation_result = str(orientation_npz)
                        sample.status = "success"
                        sample.reason = "SP and orientation features completed"
                        sample.audit["orientation"] = "present"
                    else:
                        sample.status = "failed"
                        sample.reason = "orientation extraction failed"
                        sample.audit["orientation"] = f"missing:{orientation_npz}"
                    save_sample_record(temp_run_dir, sample)

        summary_path = write_summary(temp_run_dir, samples)
        logger.info("SUMMARY | %s", summary_path)
        logger.info("LOG    | %s", main_log_path)

        success_count = sum(1 for sample in samples if sample.status == "success")
        failure_count = sum(1 for sample in samples if sample.status == "failed")
        dry_run_count = sum(1 for sample in samples if sample.status == "dry_run")
        logger.info(
            "DONE   | success=%d failed=%d dry_run=%d total=%d",
            success_count,
            failure_count,
            dry_run_count,
            len(samples),
        )
        return 0 if failure_count == 0 else 1

    except Exception as exc:
        logger.exception("FATAL  | %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
