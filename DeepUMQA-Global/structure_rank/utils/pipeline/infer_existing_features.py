#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using already prepared features only.")
    parser.add_argument("--test_fpath", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--method_type", required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=9999)
    parser.add_argument("--gpu-max-length", type=int, default=1500)
    parser.add_argument("--interface_pdb_base_dir", required=True)
    parser.add_argument("--feature_root", default="")
    parser.add_argument("--interface_fe_base_dir", default="")
    parser.add_argument("--three_di_base_dir", default="")
    parser.add_argument("--voro_base_dir", default="")
    parser.add_argument("--mpnn_base_dir", default="")
    parser.add_argument("--structure_profile_dir", default="")
    return parser.parse_args()


def validate_sample_shapes(data):
    lengths = {}

    def record(key, value):
        if value is not None:
            lengths[key] = int(value)

    if "_1d" in data:
        record("_1d", data["_1d"].shape[-1])
    if "_2d" in data:
        record("_2d_h", data["_2d"].shape[1])
        record("_2d_w", data["_2d"].shape[2])
    if "feat" in data:
        record("feat", data["feat"].shape[1])
    if "mpnn" in data:
        record("mpnn", data["mpnn"].shape[1])
    if "profile" in data:
        record("profile_h", data["profile"].shape[1])
        record("profile_w", data["profile"].shape[2])
    if "entropy" in data:
        record("entropy_h", data["entropy"].shape[1])
        record("entropy_w", data["entropy"].shape[2])
    if "af_orientation" in data:
        record("af_orientation_h", data["af_orientation"].shape[1])
        record("af_orientation_w", data["af_orientation"].shape[2])
    if "mask_3d" in data:
        record("mask_3d_h", data["mask_3d"].shape[1])
        record("mask_3d_w", data["mask_3d"].shape[2])
    if "voro_features" in data:
        record("voro_features", data["voro_features"].shape[-1])
    if "voro_normal" in data:
        record("voro_normal", data["voro_normal"].shape[-1])

    canonical = sorted(set(lengths.values()))
    if len(canonical) <= 1:
        return True, ""
    detail = ", ".join(f"{k}={v}" for k, v in sorted(lengths.items()))
    return False, detail


def move_required_tensors(data, required_keys, device):
    for key in required_keys:
        if key in data:
            data[key] = data[key].to(device)
    return data


def sample_length(data):
    if "_1d" in data:
        return int(data["_1d"].shape[-1])
    if "feat" in data:
        return int(data["feat"].shape[-1])
    raise KeyError("cannot determine sample length from batch")


def should_retry_without_cudnn(exc):
    message = str(exc).lower()
    if "out of memory" in message:
        return True
    return "cudnn" in message and "algorithm" in message


def resolve_feature_subdir(path_value, expected_name):
    if not path_value:
        return ""
    path = Path(path_value).resolve()
    direct_hit = path / expected_name
    if direct_hit.is_dir():
        return str(direct_hit)
    return str(path)


def resolve_checkpoint_paths(ckpt_path):
    ckpt_path = Path(ckpt_path).resolve()
    if ckpt_path.is_file():
        if ckpt_path.suffix.lower() in {".ckpt", ".pt", ".pth"}:
            return [ckpt_path]
        paths = []
        for raw_line in ckpt_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            candidate = Path(line)
            if not candidate.is_absolute():
                candidate = (ckpt_path.parent / candidate).resolve()
            if not candidate.exists():
                raise FileNotFoundError(f"checkpoint 不存在: {candidate}")
            paths.append(candidate)
        if paths:
            return paths
        raise RuntimeError(f"未在 {ckpt_path} 中解析到 checkpoint")
    if ckpt_path.is_dir():
        paths = sorted(path.resolve() for path in ckpt_path.iterdir() if path.suffix.lower() in {".ckpt", ".pt", ".pth"})
        if paths:
            return paths
        raise RuntimeError(f"目录中没有 checkpoint: {ckpt_path}")
    raise FileNotFoundError(f"ckpt_path 不存在: {ckpt_path}")


def sample_parts(sample_name):
    normalized = str(sample_name).strip().replace("\\", "/")
    if "/" in normalized:
        return tuple(normalized.split("/", 1))
    return "", normalized


def append_prediction_row(path, target, decoy, sample_name, score, ckpt_name="", write_header=False):
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["target", "decoy", "name", "score", "checkpoint"])
        elif sample_name:
            writer.writerow([target, decoy, sample_name, score, ckpt_name])


def main():
    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from structure_rank.Data.dataset import DecoyDataset
    from structure_rank.Model.ModelNet import ModelRank

    args = parse_args()
    start = time.time()
    gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    print("device:", gpu_device, flush=True)

    if not args.feature_root:
        args.feature_root = os.path.join(os.path.dirname(args.interface_pdb_base_dir.rstrip("/")), "feature")
    args.feature_root = str(Path(args.feature_root).resolve())
    args.interface_fe_base_dir = resolve_feature_subdir(args.interface_fe_base_dir or args.feature_root, "base")
    args.three_di_base_dir = resolve_feature_subdir(args.three_di_base_dir or args.feature_root, "3di")
    args.voro_base_dir = resolve_feature_subdir(args.voro_base_dir or args.feature_root, "voro")
    args.mpnn_base_dir = resolve_feature_subdir(args.mpnn_base_dir or args.feature_root, "mpnn")
    args.structure_profile_dir = resolve_feature_subdir(args.structure_profile_dir or args.feature_root, "ori")
    print(f"[INFO] Feature root: {args.feature_root}", flush=True)
    print(f"[INFO] Base feature dir: {args.interface_fe_base_dir}", flush=True)
    print(f"[INFO] 3di dir: {args.three_di_base_dir}", flush=True)
    print(f"[INFO] Voro dir: {args.voro_base_dir}", flush=True)
    print(f"[INFO] MPNN dir: {args.mpnn_base_dir}", flush=True)
    print(f"[INFO] Profile dir: {args.structure_profile_dir}", flush=True)

    dataset = DecoyDataset(
        targets_fpath=args.test_fpath,
        process_feat=False,
        max_length=args.max_length,
        interface_pdb_base_dir=args.interface_pdb_base_dir,
        feature_root=args.feature_root,
        interface_fe_base_dir=args.interface_fe_base_dir,
        three_di_base_dir=args.three_di_base_dir,
        voro_base_dir=args.voro_base_dir,
        mpnn_base_dir=args.mpnn_base_dir,
        structure_profile_dir=args.structure_profile_dir,
        infer=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    ckpt_paths = resolve_checkpoint_paths(args.ckpt_path)
    if not ckpt_paths:
        print(f"[ERROR] No valid checkpoints found at: {args.ckpt_path}", flush=True)
        return 1

    model = ModelRank()
    score_dir = os.path.join(args.output, "SCORE_interface")
    os.makedirs(score_dir, exist_ok=True)

    for ckpt_path in ckpt_paths:
        ckpt_name = os.path.basename(ckpt_path)
        ckpt_index = ckpt_name.split("-")[0]
        out_path = os.path.join(score_dir, f"pred.{args.method_type}.{ckpt_index}.csv")
        if os.path.exists(out_path):
            os.remove(out_path)
        append_prediction_row(out_path, "", "", "", "", "", write_header=True)

        print(f"[INFO] Loading checkpoint: {ckpt_path}", flush=True)
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {k[6:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        gpu_model = None
        cpu_model = None
        cudnn_disabled = False
        if gpu_device.type == "cuda":
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False

        def get_model(run_device):
            nonlocal gpu_model, cpu_model
            if run_device.type == "cuda":
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

        success = 0
        skipped = 0
        required_keys = [
            "_1d", "_2d", "vidx", "val", "feat", "adj",
            "voro_features", "voro_normal", "profile",
            "entropy", "mask_3d", "af_orientation", "mpnn",
        ]
        with torch.no_grad():
            for data in tqdm(dataloader, desc=ckpt_index):
                sample_name = "Unknown"
                if isinstance(data, dict) and "name" in data and len(data["name"]) > 0:
                    sample_name = data["name"][0]
                target, decoy = sample_parts(sample_name)
                try:
                    if data is None or not isinstance(data, dict):
                        skipped += 1
                        continue
                    if len(list(data.keys())) <= 1:
                        print(f"[WARN] Missing features for {sample_name}, skipping.", flush=True)
                        skipped += 1
                        continue
                    ok, reason = validate_sample_shapes(data)
                    if not ok:
                        print(f"[WARN] Shape mismatch for {sample_name}: {reason}", flush=True)
                        skipped += 1
                        continue

                    length = sample_length(data)
                    run_device = cpu_device if length > args.gpu_max_length else gpu_device
                    model_for_sample = get_model(run_device)
                    move_required_tensors(data, required_keys, run_device)
                    output = model_for_sample(data)
                    score = float(output.detach().reshape(-1)[0].item())
                    append_prediction_row(out_path, target, decoy, sample_name, score, ckpt_name)
                    success += 1
                    print(f"[INFO] {target}/{decoy} -> {score:.6f} [{run_device.type}]", flush=True)
                except RuntimeError as exc:
                    retry_on_cpu = False
                    if run_device.type == "cuda" and should_retry_without_cudnn(exc):
                        if not cudnn_disabled:
                            cudnn_disabled = True
                            torch.backends.cudnn.enabled = False
                            torch.cuda.empty_cache()
                            print(
                                f"[WARN] cuDNN failed for {sample_name}; retrying on GPU with cuDNN disabled.",
                                flush=True,
                            )
                            try:
                                output = model_for_sample(data)
                                score = float(output.detach().reshape(-1)[0].item())
                                append_prediction_row(out_path, target, decoy, sample_name, score, ckpt_name)
                                success += 1
                                print(f"[INFO] {target}/{decoy} -> {score:.6f} [cuda-nocudnn]", flush=True)
                                continue
                            except RuntimeError as retry_exc:
                                exc = retry_exc
                        retry_on_cpu = True

                    if run_device.type == "cuda" and retry_on_cpu:
                        try:
                            cpu_model = get_model(cpu_device)
                            move_required_tensors(data, required_keys, cpu_device)
                            output = cpu_model(data)
                            score = float(output.detach().reshape(-1)[0].item())
                            append_prediction_row(out_path, target, decoy, sample_name, score, ckpt_name)
                            success += 1
                            print(f"[INFO] {target}/{decoy} -> {score:.6f} [cpu-fallback]", flush=True)
                            continue
                        except Exception as cpu_exc:
                            exc = cpu_exc

                    if "out of memory" in str(exc).lower() and gpu_device.type == "cuda":
                        print(f"[OOM] {sample_name}", flush=True)
                        torch.cuda.empty_cache()
                    else:
                        print(f"[ERROR] Runtime error for {sample_name}: {exc}", flush=True)
                        traceback.print_exc()
                    skipped += 1
                except Exception as exc:
                    print(f"[FAIL] {sample_name}: {exc}", flush=True)
                    traceback.print_exc()
                    skipped += 1

        print(f"[INFO] Finished {ckpt_name}: success={success} skipped={skipped} csv={out_path}", flush=True)

    print(f"[INFO] Total runtime: {time.time() - start:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
