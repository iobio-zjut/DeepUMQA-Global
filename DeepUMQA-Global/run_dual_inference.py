#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


METHOD_GLOBAL = "DeepUMQA-G_V2_global"
METHOD_INTERFACE = "DeepUMQA-G_V2_interface"
STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}


@dataclass(frozen=True)
class SampleSpec:
    target: str
    model_name: str
    source_path: Path

    @property
    def model_base(self) -> str:
        return self.model_name[:-4] if self.model_name.lower().endswith(".pdb") else Path(self.model_name).stem

    @property
    def output_name(self) -> str:
        return f"{self.target}/{self.model_name}"


@dataclass(frozen=True)
class SampleResult:
    sample: SampleSpec
    global_score: float
    interface_score: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Single-machine DeepUMQA-G entrypoint for feature generation and global/interface inference."
    )
    parser.add_argument("--pdb-root", default="./example/pdb", help="Input decoy root directory")
    parser.add_argument("--query-root", default="./example/query", help="Input query/reference root directory")
    parser.add_argument("--feature-root", default="./example/feature", help="Feature cache directory")
    parser.add_argument("--output-root", default="./example/output", help="Final output directory")
    parser.add_argument("--ckpt-path", default="./checkpoints", help="Checkpoint file or directory")
    parser.add_argument("--pdb-list", default="", help="Optional file listing PDB inputs")
    parser.add_argument("--ckpt-list", default="", help="Optional file listing checkpoints")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for subprocess stages")
    parser.add_argument(
        "--foldseek-bin",
        default=os.environ.get("DEEPUMQA_FOLDSEEK_BIN", ""),
        help="Foldseek executable path; defaults to DEEPUMQA_FOLDSEEK_BIN or PATH",
    )
    parser.add_argument(
        "--mpnn-python",
        default=os.environ.get("DEEPUMQA_MPNN_PYTHON", sys.executable),
        help="ProteinMPNN Python executable; defaults to DEEPUMQA_MPNN_PYTHON or current Python",
    )
    parser.add_argument(
        "--voro-python",
        default=os.environ.get("DEEPUMQA_VORO_PYTHON", sys.executable),
        help="Voronota helper Python executable; defaults to DEEPUMQA_VORO_PYTHON or current Python",
    )
    parser.add_argument(
        "--pyrosetta-python",
        default=os.environ.get("DEEPUMQA_PYROSETTA_PYTHON", sys.executable),
        help="PyRosetta Python executable; defaults to DEEPUMQA_PYROSETTA_PYTHON or current Python",
    )
    parser.add_argument(
        "--voro-exe-dir",
        default=os.environ.get("DEEPUMQA_VORO_EXE_DIR", ""),
        help="Optional Voronota executable directory",
    )
    parser.add_argument(
        "--sp-template-db",
        default=os.environ.get("DEEPUMQA_SP_TEMPLATE_DB", ""),
        help="Optional Foldseek template database for multimer SP",
    )
    parser.add_argument(
        "--sp-monomer-template-db",
        default=os.environ.get("DEEPUMQA_SP_MONOMER_TEMPLATE_DB", ""),
        help="Optional Foldseek template database for monomer SP",
    )
    parser.add_argument(
        "--afdb-dir",
        default=os.environ.get("DEEPUMQA_AFDB_DIR", ""),
        help="Optional AFDB template directory for monomer SP extraction",
    )
    parser.add_argument("--force", action="store_true", help="Force rebuilding features and predictions")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary run directories under feature-root/.runs")
    parser.add_argument("--feature-workers", type=int, default=4, help="Workers for full-length feature extraction")
    parser.add_argument("--interface-workers", type=int, default=4, help="Workers for interface cropping")
    parser.add_argument("--infer-workers", type=int, default=1, help="DataLoader workers for inference")
    parser.add_argument("--cb-cutoff", type=float, default=8.0, help="Interface cutoff in angstrom")
    parser.add_argument(
        "--skip-feature-generation",
        action="store_true",
        help="Skip full-length feature generation and reuse existing features",
    )
    parser.add_argument("--gpu-max-length", type=int, default=1500, help="Length threshold for CPU inference fallback")
    parser.add_argument("--max-length", type=int, default=9999, help="Maximum sequence length passed to the dataset")
    parser.add_argument(
        "--feature-pipeline-script",
        default=str(repo_root / "structure_rank" / "utils" / "pipeline" / "feature_pipeline.py"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--interface-pipeline-script",
        default=str(repo_root / "structure_rank" / "utils" / "pipeline" / "interface_feature_pipeline.py"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def summarize_text(text: str, limit: int = 600) -> str:
    compact = " ".join((text or "").splitlines()).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]} ...[truncated]"


def log_status(prefix: str, sample_tag: str, message: str) -> None:
    print(f"[{prefix}] {sample_tag} | {message}", flush=True)


def write_stage_log(path: Path, cmd: Sequence[object], completed: subprocess.CompletedProcess[str]) -> None:
    ensure_dir(path.parent)
    lines = [
        f"CMD: {' '.join(str(item) for item in cmd)}",
        f"RETURN_CODE: {completed.returncode}",
        "",
        "[STDOUT]",
        completed.stdout or "",
        "",
        "[STDERR]",
        completed.stderr or "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_command(
    cmd: Sequence[object],
    *,
    cwd: Path,
    stage_name: str,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
) -> None:
    completed = subprocess.run(
        [str(item) for item in cmd],
        cwd=str(cwd),
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    write_stage_log(log_path, cmd, completed)
    if completed.returncode != 0:
        detail = summarize_text(completed.stderr or completed.stdout)
        raise RuntimeError(f"{stage_name} failed: {detail or f'exit code {completed.returncode}'}")


def parse_pipeline_log_entry(line: str) -> Optional[List[str]]:
    parts = [part.strip() for part in line.strip().split(" | ")]
    return parts if len(parts) >= 4 else None


def read_log_slice(log_path: Path, start_line: int = 0) -> List[str]:
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[start_line:] if start_line > 0 else lines


def format_feature_progress(parts: List[str], target_name: str, model_base: str) -> Optional[str]:
    if len(parts) < 5:
        return None
    record_type = parts[1]
    if record_type == "TARGET" and parts[2] == target_name:
        stage = parts[3]
        status = parts[4].lower()
        if stage == "SP_ORI" and status == "start":
            return "sp/ori started"
        if stage == "SP_ORI" and status == "skipped":
            return "sp/ori reused"
        if stage in {"sp", "ori"}:
            return f"{stage} {status}"
        return None
    if record_type == "DECOY" and parts[2] == model_base:
        stage = parts[3]
        status = parts[4].lower()
        if stage == "START":
            return "decoy features started"
        if stage in {"base", "3di", "voro", "mpnn", "consistency"}:
            return f"{stage} {status}"
    return None


def stream_feature_pipeline(
    cmd: Sequence[object],
    *,
    cwd: Path,
    log_path: Path,
    feature_log_path: Path,
    target_name: str,
    model_base: str,
    sample_tag: str,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        [str(item) for item in cmd],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    seen_messages: set[str] = set()
    initial_feature_log_line = len(read_log_slice(feature_log_path, 0))
    feature_log_line = initial_feature_log_line

    def drain_feature_log() -> None:
        nonlocal feature_log_line
        new_lines = read_log_slice(feature_log_path, feature_log_line)
        for raw_line in new_lines:
            parts = parse_pipeline_log_entry(raw_line)
            if not parts:
                continue
            message = format_feature_progress(parts, target_name, model_base)
            if message and message not in seen_messages:
                seen_messages.add(message)
                log_status("FEATURE", sample_tag, message)
        feature_log_line += len(new_lines)

    while True:
        drain_feature_log()
        if process.poll() is not None:
            break
        time.sleep(1)

    stdout, stderr = process.communicate()
    drain_feature_log()
    completed = subprocess.CompletedProcess(
        args=[str(item) for item in cmd],
        returncode=process.returncode or 0,
        stdout=stdout,
        stderr=stderr,
    )
    write_stage_log(log_path, cmd, completed)
    completed.start_line = initial_feature_log_line  # type: ignore[attr-defined]
    return completed


def collect_feature_status_summary(log_path: Path, target_name: str, model_base: str, start_line: int = 0) -> str:
    target_statuses: Dict[str, Tuple[str, str]] = {}
    decoy_statuses: Dict[str, Tuple[str, str]] = {}
    for raw_line in read_log_slice(log_path, start_line):
        parts = parse_pipeline_log_entry(raw_line)
        if not parts or len(parts) < 5:
            continue
        record_type = parts[1]
        subject = parts[2]
        stage = parts[3]
        status = parts[4].lower()
        detail = parts[5] if len(parts) > 5 else ""
        if record_type == "TARGET" and subject == target_name:
            if stage in {"sp", "ori"}:
                target_statuses[stage] = (status, detail)
            elif stage == "SP_ORI" and status in {"failed", "skipped"}:
                target_statuses["sp"] = (status, detail)
                target_statuses["ori"] = (status, detail)
        if record_type == "DECOY" and subject == model_base and stage in {"base", "3di", "voro", "mpnn", "consistency"}:
            decoy_statuses[stage] = (status, detail)

    def format_section(name: str, statuses: Dict[str, Tuple[str, str]], order: List[str]) -> Optional[str]:
        bits = []
        for key in order:
            if key not in statuses:
                continue
            status, detail = statuses[key]
            if detail and status == "failed":
                bits.append(f"{key}={status}({summarize_text(detail, 120)})")
            else:
                bits.append(f"{key}={status}")
        return f"{name}[{', '.join(bits)}]" if bits else None

    sections = []
    target_section = format_section("target", target_statuses, ["sp", "ori"])
    decoy_section = format_section("decoy", decoy_statuses, ["base", "3di", "voro", "mpnn", "consistency"])
    if target_section:
        sections.append(target_section)
    if decoy_section:
        sections.append(decoy_section)
    return "; ".join(sections) if sections else "no parsed feature status"


def stage_input_file(source: Path, destination: Path, force: bool) -> None:
    ensure_dir(destination.parent)
    if destination.exists():
        if not force:
            return
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    shutil.copy2(source, destination)


def has_full_length_feature_bundle(feature_root: Path, target: str, model_base: str) -> bool:
    required = [
        feature_root / "base" / target / f"{model_base}.npz",
        feature_root / "3di" / target / f"{model_base}.npz",
        feature_root / "voro" / target / f"{model_base}.npz",
        feature_root / "mpnn" / target / f"{model_base}.npz",
        feature_root / "ori" / f"{target}.npz",
    ]
    return all(path.exists() for path in required)


def write_list_file(path: Path, sample_path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(f"{sample_path}\n", encoding="utf-8")


def resolve_path_from_list(base_dir: Path, list_file: Path, raw_value: str) -> Path:
    raw_path = Path(raw_value)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((list_file.parent / raw_path).resolve())
        candidates.append((base_dir / raw_path).resolve())
        if raw_path.suffix.lower() != ".pdb":
            candidates.append((list_file.parent / raw_path).with_suffix(".pdb").resolve())
            candidates.append((base_dir / raw_path).with_suffix(".pdb").resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"entry from {list_file} not found: {raw_value}")


def infer_target_from_path(pdb_root: Path, pdb_path: Path) -> str:
    try:
        rel = pdb_path.resolve().relative_to(pdb_root.resolve())
        if len(rel.parts) >= 2:
            return rel.parts[-2]
    except Exception:
        pass
    return pdb_path.stem


def is_structure_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in STRUCTURE_SUFFIXES and not path.name.startswith(".")


def discover_samples(pdb_root: Path, pdb_list: Optional[Path]) -> List[SampleSpec]:
    samples: List[SampleSpec] = []
    if pdb_list:
        for raw_line in pdb_list.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            pdb_path = resolve_path_from_list(pdb_root, pdb_list, line)
            if pdb_path.suffix.lower() != ".pdb":
                raise ValueError(f"only PDB decoys are supported here: {pdb_path}")
            samples.append(
                SampleSpec(
                    target=infer_target_from_path(pdb_root, pdb_path),
                    model_name=pdb_path.name,
                    source_path=pdb_path.resolve(),
                )
            )
        if not samples:
            raise RuntimeError(f"no PDB inputs found in list: {pdb_list}")
        return sorted(samples, key=lambda item: (item.target, item.model_name))

    if not pdb_root.exists():
        raise FileNotFoundError(f"pdb-root does not exist: {pdb_root}")
    for pdb_path in sorted(pdb_root.rglob("*.pdb")):
        if not is_structure_file(pdb_path):
            continue
        samples.append(
            SampleSpec(
                target=infer_target_from_path(pdb_root, pdb_path),
                model_name=pdb_path.name,
                source_path=pdb_path.resolve(),
            )
        )
    if not samples:
        raise RuntimeError(f"no PDB files found under: {pdb_root}")
    return samples


def build_query_index(query_root: Path) -> Dict[str, Path]:
    if not query_root.exists():
        raise FileNotFoundError(f"query-root does not exist: {query_root}")
    index: Dict[str, Path] = {}
    for path in sorted(query_root.rglob("*")):
        if not is_structure_file(path):
            continue
        for key in {path.stem, path.parent.name}:
            if key and key not in index:
                index[key] = path.resolve()
    return index


def resolve_query_path(target: str, query_index: Dict[str, Path]) -> Optional[Path]:
    candidates = [target, target.upper(), target.lower()]
    for candidate in candidates:
        if candidate in query_index:
            return query_index[candidate]
    return None


def resolve_checkpoint_paths(ckpt_path: Path, ckpt_list: Optional[Path]) -> List[Path]:
    checkpoints: List[Path] = []
    if ckpt_list:
        for raw_line in ckpt_list.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            candidate = Path(line)
            if not candidate.is_absolute():
                candidate = (ckpt_list.parent / candidate).resolve()
            if not candidate.exists():
                raise FileNotFoundError(f"checkpoint listed in {ckpt_list} not found: {line}")
            checkpoints.append(candidate)
    elif ckpt_path.is_file():
        if ckpt_path.suffix.lower() not in {".ckpt", ".pt", ".pth"}:
            raise ValueError(f"unsupported checkpoint file: {ckpt_path}")
        checkpoints.append(ckpt_path.resolve())
    elif ckpt_path.is_dir():
        checkpoints.extend(sorted(path.resolve() for path in ckpt_path.iterdir() if path.suffix.lower() in {".ckpt", ".pt", ".pth"}))
    else:
        raise FileNotFoundError(f"ckpt-path does not exist: {ckpt_path}")

    if not checkpoints:
        source = ckpt_list if ckpt_list else ckpt_path
        raise RuntimeError(f"no .ckpt/.pt/.pth files found in: {source}")
    return checkpoints


def resolve_feature_dir(path_value: str, expected_name: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value).resolve()
    direct_hit = path / expected_name
    if direct_hit.is_dir():
        return str(direct_hit)
    return str(path)


def collect_scores(score_dir: Path) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for csv_path in sorted(score_dir.glob("pred.*.csv")):
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                scores[csv_path.name] = float(row["score"])
                break
    if not scores:
        raise RuntimeError(f"no prediction CSV files found in {score_dir}")
    return scores


def summarize_scores(scores: Dict[str, float]) -> Dict[str, object]:
    values = list(scores.values())
    return {"per_checkpoint": scores, "mean_score": sum(values) / len(values)}


def run_inference_pass(
    *,
    test_fpath: Path,
    output_root: Path,
    checkpoint_paths: Sequence[Path],
    method_type: str,
    num_workers: int,
    gpu_max_length: int,
    max_length: int,
    interface_pdb_base_dir: Path,
    feature_root: Path,
) -> Dict[str, object]:
    import torch
    from torch.utils.data import DataLoader

    from structure_rank.Data.dataset import DecoyDataset
    from structure_rank.Model.ModelNet import ModelRank

    feature_root = feature_root.resolve()
    interface_pdb_base_dir = interface_pdb_base_dir.resolve()
    output_root = output_root.resolve()
    ensure_dir(output_root)

    gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    dataset = DecoyDataset(
        targets_fpath=str(test_fpath),
        process_feat=False,
        max_length=max_length,
        interface_pdb_base_dir=str(interface_pdb_base_dir),
        feature_root=str(feature_root),
        interface_fe_base_dir=resolve_feature_dir(str(feature_root), "base"),
        three_di_base_dir=resolve_feature_dir(str(feature_root), "3di"),
        voro_base_dir=resolve_feature_dir(str(feature_root), "voro"),
        mpnn_base_dir=resolve_feature_dir(str(feature_root), "mpnn"),
        structure_profile_dir=resolve_feature_dir(str(feature_root), "ori"),
        infer=True,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=num_workers)

    for checkpoint_path in checkpoint_paths:
        ckpt_name = checkpoint_path.name
        ckpt_index = ckpt_name.split("-")[0]
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        state_dict = {key[6:]: value for key, value in state_dict.items()}
        gpu_model = None
        cpu_model = None

        def get_model(device: torch.device) -> ModelRank:
            nonlocal gpu_model, cpu_model
            if device.type == "cuda":
                if gpu_model is None:
                    gpu_model = ModelRank()
                    gpu_model.load_state_dict(state_dict)
                    gpu_model = gpu_model.to(gpu_device)
                    gpu_model.eval()
                return gpu_model
            if cpu_model is None:
                cpu_model = ModelRank()
                cpu_model.load_state_dict(state_dict)
                cpu_model = cpu_model.to(cpu_device)
                cpu_model.eval()
            return cpu_model

        score_dir = ensure_dir(output_root / "SCORE_interface")
        out_fpath = score_dir / f"pred.{method_type}.{ckpt_index}.csv"
        with out_fpath.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "score"])

        with torch.no_grad():
            for data in dataloader:
                sample_name = "Unknown"
                if isinstance(data, dict) and "name" in data and len(data["name"]) > 0:
                    sample_name = data["name"][0]
                if data is None or not isinstance(data, dict) or len(list(data.keys())) <= 1:
                    continue
                if "score" in data:
                    score = 0.0
                else:
                    required_keys = [
                        "_1d",
                        "_2d",
                        "vidx",
                        "val",
                        "feat",
                        "adj",
                        "voro_features",
                        "voro_normal",
                        "profile",
                        "entropy",
                        "mask_3d",
                        "af_orientation",
                        "mpnn",
                    ]
                    length = int(data["_1d"].shape[-1]) if "_1d" in data else int(data["feat"].shape[-1])
                    run_device = cpu_device if length > gpu_max_length else gpu_device
                    for key in required_keys:
                        if key in data:
                            data[key] = data[key].to(run_device)
                    output = get_model(run_device)(data)
                    score = float(output.detach().reshape(-1)[0].item())
                row_name = sample_name.split("/")[-1]
                with out_fpath.open("a", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow([row_name, score])

    return summarize_scores(collect_scores(output_root / "SCORE_interface"))


def write_final_csv(path: Path, rows: Sequence[Tuple[str, float]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "score"])
        for name, score in rows:
            writer.writerow([name, score])


def build_stage_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    env["DEEPUMQA_FOLDSEEK_BIN"] = args.foldseek_bin or shutil.which("foldseek") or ""
    env["DEEPUMQA_MPNN_PYTHON"] = args.mpnn_python or sys.executable
    env["DEEPUMQA_VORO_PYTHON"] = args.voro_python or sys.executable
    env["DEEPUMQA_PYROSETTA_PYTHON"] = args.pyrosetta_python or sys.executable
    if args.voro_exe_dir:
        env["DEEPUMQA_VORO_EXE_DIR"] = args.voro_exe_dir
    if args.sp_template_db:
        env["DEEPUMQA_SP_TEMPLATE_DB"] = args.sp_template_db
    if args.sp_monomer_template_db:
        env["DEEPUMQA_SP_MONOMER_TEMPLATE_DB"] = args.sp_monomer_template_db
    if args.afdb_dir:
        env["DEEPUMQA_AFDB_DIR"] = args.afdb_dir
    return env


def process_sample(
    *,
    sample: SampleSpec,
    query_path: Path,
    repo_root: Path,
    feature_root: Path,
    run_root: Path,
    checkpoints: Sequence[Path],
    args: argparse.Namespace,
    env: Dict[str, str],
) -> SampleResult:
    sample_tag = sample.output_name
    sample_root = run_root / sample.target / sample.model_name
    staged_pdb_root = sample_root / "input" / "pdb"
    staged_query_root = sample_root / "input" / "query"
    staged_pdb = staged_pdb_root / sample.target / sample.model_name
    staged_query = staged_query_root / f"{sample.target}.pdb"
    interface_root = sample_root / "interface"
    interface_pdb_root = interface_root / "pdb"
    interface_feature_root = interface_root / "feature"
    interface_mapping_root = interface_root / "mapping"
    stage_log_root = ensure_dir(sample_root / "stage_logs")
    feature_log_path = feature_root / "logs" / f"{sample.target}.log"
    error_log_path = sample_root / "error.log"

    try:
        log_status("PROCESS", sample_tag, "started")
        stage_input_file(sample.source_path, staged_pdb, args.force)
        stage_input_file(query_path, staged_query, args.force)

        if not args.skip_feature_generation:
            if not has_full_length_feature_bundle(feature_root, sample.target, sample.model_base) or args.force:
                feature_cmd: List[object] = [
                    args.python_bin,
                    Path(args.feature_pipeline_script),
                    "--pdb-root",
                    staged_pdb_root,
                    "--query-root",
                    staged_query_root,
                    "--feature-root",
                    feature_root,
                    "--workers",
                    str(args.feature_workers),
                    "--targets",
                    sample.target,
                ]
                if args.force:
                    feature_cmd.append("--force")
                completed = stream_feature_pipeline(
                    feature_cmd,
                    cwd=repo_root,
                    log_path=stage_log_root / "feature.log",
                    feature_log_path=feature_log_path,
                    target_name=sample.target,
                    model_base=sample.model_base,
                    sample_tag=sample_tag,
                    env=env,
                )
                if completed.returncode != 0:
                    feature_summary = collect_feature_status_summary(
                        feature_log_path,
                        sample.target,
                        sample.model_base,
                        getattr(completed, "start_line", 0),
                    )
                    raise RuntimeError(f"feature extraction failed | {feature_summary}")
            else:
                log_status("FEATURE", sample_tag, "success (reuse full-length features)")
        elif not has_full_length_feature_bundle(feature_root, sample.target, sample.model_base):
            raise RuntimeError("skip-feature-generation was set but required full-length features are missing")

        interface_cmd: List[object] = [
            args.python_bin,
            Path(args.interface_pipeline_script),
            "--pdb-root",
            staged_pdb_root,
            "--feature-root",
            feature_root,
            "--output-pdb-root",
            interface_pdb_root,
            "--output-feature-root",
            interface_feature_root,
            "--mapping-root",
            interface_mapping_root,
            "--workers",
            str(args.interface_workers),
            "--cb-cutoff",
            str(args.cb_cutoff),
            "--targets",
            sample.target,
        ]
        if args.force:
            interface_cmd.append("--force")
        run_command(
            interface_cmd,
            cwd=repo_root,
            stage_name="interface feature extraction",
            log_path=stage_log_root / "interface_feature.log",
            env=env,
        )
        log_status("FEATURE", sample_tag, "success")

        staged_interface_pdb = interface_pdb_root / sample.target / sample.model_name
        if not staged_interface_pdb.exists():
            raise RuntimeError(f"interface PDB was not generated: {staged_interface_pdb}")

        global_list = sample_root / "lists" / "global.txt"
        interface_list = sample_root / "lists" / "interface.txt"
        write_list_file(global_list, staged_pdb)
        write_list_file(interface_list, staged_interface_pdb)

        global_scores = run_inference_pass(
            test_fpath=global_list,
            output_root=sample_root / "predictions" / "global",
            checkpoint_paths=checkpoints,
            method_type=METHOD_GLOBAL,
            num_workers=args.infer_workers,
            gpu_max_length=args.gpu_max_length,
            max_length=args.max_length,
            interface_pdb_base_dir=staged_pdb_root,
            feature_root=feature_root,
        )
        run_command(
            [args.python_bin, "-c", "print('global inference completed')"],
            cwd=repo_root,
            stage_name="global inference marker",
            log_path=stage_log_root / "global_infer.log",
        )

        interface_scores = run_inference_pass(
            test_fpath=interface_list,
            output_root=sample_root / "predictions" / "interface",
            checkpoint_paths=checkpoints,
            method_type=METHOD_INTERFACE,
            num_workers=args.infer_workers,
            gpu_max_length=args.gpu_max_length,
            max_length=args.max_length,
            interface_pdb_base_dir=interface_pdb_root,
            feature_root=interface_feature_root,
        )
        run_command(
            [args.python_bin, "-c", "print('interface inference completed')"],
            cwd=repo_root,
            stage_name="interface inference marker",
            log_path=stage_log_root / "interface_infer.log",
        )

        shutil.rmtree(sample_root / "predictions", ignore_errors=True)
        log_status(
            "INFER",
            sample_tag,
            f"global={global_scores['mean_score']:.6f}, interface={interface_scores['mean_score']:.6f}",
        )
        return SampleResult(
            sample=sample,
            global_score=float(global_scores["mean_score"]),
            interface_score=float(interface_scores["mean_score"]),
        )
    except Exception:
        error_log_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    pdb_root = Path(args.pdb_root).resolve()
    query_root = Path(args.query_root).resolve()
    feature_root = ensure_dir(Path(args.feature_root).resolve())
    output_root = ensure_dir(Path(args.output_root).resolve())
    ckpt_path = Path(args.ckpt_path).resolve()
    pdb_list = Path(args.pdb_list).resolve() if args.pdb_list else None
    ckpt_list = Path(args.ckpt_list).resolve() if args.ckpt_list else None
    env = build_stage_env(args)
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    run_root = ensure_dir(feature_root / ".runs" / run_id)

    try:
        samples = discover_samples(pdb_root, pdb_list)
        query_index = build_query_index(query_root)
        checkpoints = resolve_checkpoint_paths(ckpt_path, ckpt_list)

        global_rows: List[Tuple[str, float]] = []
        interface_rows: List[Tuple[str, float]] = []
        failures: List[str] = []

        for sample in samples:
            query_path = resolve_query_path(sample.target, query_index)
            if query_path is None:
                failures.append(f"{sample.output_name}: query structure not found under {query_root}")
                continue
            try:
                result = process_sample(
                    sample=sample,
                    query_path=query_path,
                    repo_root=repo_root,
                    feature_root=feature_root,
                    run_root=run_root,
                    checkpoints=checkpoints,
                    args=args,
                    env=env,
                )
            except Exception as exc:
                failures.append(f"{sample.output_name}: {summarize_text(str(exc), 240)}")
                log_status("ERROR", sample.output_name, summarize_text(str(exc), 240))
                continue

            global_rows.append((result.sample.output_name, result.global_score))
            interface_rows.append((result.sample.output_name, result.interface_score))

        write_final_csv(output_root / "global_score.csv", global_rows)
        write_final_csv(output_root / "interface_score.csv", interface_rows)

        for path in sorted(output_root.iterdir()):
            if path.is_file() and path.name not in {"global_score.csv", "interface_score.csv"}:
                path.unlink()
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

        if failures:
            (run_root / "failures.log").write_text("\n".join(failures) + "\n", encoding="utf-8")
            return 1

        if not args.keep_temp:
            shutil.rmtree(run_root, ignore_errors=True)
        return 0
    except Exception as exc:
        ensure_dir(run_root)
        (run_root / "fatal.log").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
