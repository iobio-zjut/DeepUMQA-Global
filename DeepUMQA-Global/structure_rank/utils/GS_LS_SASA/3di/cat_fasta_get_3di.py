#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------------------------
# 3Di AA mapping
# -------------------------
SEQ3DI = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
          "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
SEQ3DIMAP = {aa: i for i, aa in enumerate(SEQ3DI)}

def one_hot_encode(sequence: str) -> np.ndarray:
    one_hot = np.zeros((20, len(sequence)), dtype=np.float32)
    for i, aa in enumerate(sequence):
        j = SEQ3DIMAP.get(aa, None)
        if j is not None:
            one_hot[j, i] = 1.0
    return one_hot

def get_chain_names(pdb_file: str):
    chain_names = set()
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                if len(line) > 21:
                    c = line[21].strip()
                    if c: chain_names.add(c)
    return sorted(list(chain_names))

def build_concat_sequence_for_one_pdb(pdb_path: str, fasta_sequences: dict):
    pdb_file = os.path.basename(pdb_path)
    model_name = os.path.splitext(pdb_file)[0]
    chain_names = get_chain_names(pdb_path)
    concat = []
    all_keys = list(fasta_sequences.keys())

    for idx, chain in enumerate(chain_names):
        cands = [f"{model_name}_{chain}", f"{pdb_file}_{chain}", f"{model_name}{chain}", f"{pdb_file} {idx}"]
        found_seq = None
        for cand in cands:
            if cand in fasta_sequences:
                found_seq = str(fasta_sequences[cand].seq)
                break
        
        if not found_seq: # 模糊匹配
            for fid in all_keys:
                if model_name in fid and (chain in fid or str(idx) in fid):
                    found_seq = str(fasta_sequences[fid].seq)
                    break

        if found_seq:
            concat.append(found_seq)
        else:
            continue

    return "".join(concat)

def process_one_model(task):
    target, pdb_path, fasta_subdir, out_npz_subdir = task
    model_prefix = os.path.splitext(os.path.basename(pdb_path))[0]

    # 寻找匹配的 FASTA
    fasta_file_path = None
    for fn in [f"{model_prefix}.fasta", f"{model_prefix}_3di.fasta"]:
        p = os.path.join(fasta_subdir, fn)
        if os.path.exists(p):
            fasta_file_path = p
            break

    if not fasta_file_path:
        return (False, f"⚠️ No FASTA for {model_prefix}")

    # 读取并处理
    try:
        fasta_sequences = {r.id: r for r in SeqIO.parse(fasta_file_path, "fasta")}
        concat_seq = build_concat_sequence_for_one_pdb(pdb_path, fasta_sequences)
        
        if concat_seq:
            os.makedirs(out_npz_subdir, exist_ok=True)
            np.savez(os.path.join(out_npz_subdir, f"{model_prefix}.npz"), seq3di=one_hot_encode(concat_seq))
            # 核心优化：处理完一个就删一个
            os.remove(fasta_file_path)
            return (True, f"✅ {model_prefix} Done")
        else:
            return (False, f"❌ {model_prefix} Seq Empty")
    except Exception as e:
        return (False, f"❌ {model_prefix} Error: {str(e)}")

def main():
    pdb_dir = os.environ.get("PDB_BASE")
    fasta_base_dir = os.environ.get("SEQ3DI_BASE")
    output_npz_base_dir = os.environ.get("OUT_NPZ_BASE")
    max_workers = int(os.environ.get("MAX_JOBS", 4))

    tasks = []
    for root, _, files in os.walk(pdb_dir):
        for fn in files:
            if not fn.endswith(".pdb"): continue
            pdb_path = os.path.join(root, fn)
            rel = os.path.relpath(pdb_path, pdb_dir)
            target = rel.split(os.sep)[0]
            tasks.append((target, pdb_path, os.path.join(fasta_base_dir, target), os.path.join(output_npz_base_dir, target)))

    print(f"[INFO] Processing {len(tasks)} models...")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(process_one_model, t) for t in tasks]
        for fut in as_completed(futs):
            success, msg = fut.result()
            if not success: print(msg)

if __name__ == "__main__":
    main()