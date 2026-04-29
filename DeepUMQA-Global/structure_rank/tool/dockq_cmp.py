import sys
import os

env_path = "%s/../../" % os.path.dirname(os.path.abspath(__file__))
sys.path.append(env_path)

import argparse
import numpy as np
import itertools

from structure_rank.tool.utils import tmpdir_manager
from structure_rank.tool.utils import is_antibody
from structure_rank.tool.utils import parse_pdb
from structure_rank.tool.utils import get_similarity

from structure_rank.tool.logger import Logger

logger = Logger.logger


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdb_paths_1", type=str, default=None, help="pdb path")
    parser.add_argument("--pdb_paths_2", type=str, default=None, help="pdb path")
    parser.add_argument(
        "--max_dockq",
        action="store_true",
        default=False,
        help="compute max dockq or not",
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        default=None,
        help="path of result csv (optional)",
    )
    args = parser.parse_args()

    return args


def get_fasta_file_list(pdb_paths, postfix="pdb"):
    if pdb_paths.endswith(postfix):
        pdb_list = [pdb_paths]
    else:
        pdb_list = np.atleast_1d(np.loadtxt(pdb_paths, dtype="str"))

    return pdb_list


def evaluation(results):
    metrics = dict()
    metrics["DockQ"] = []
    for ret in results:
        metrics["DockQ"].append(np.array(list(ret.values())).mean())

    mu_std_metrics = dict()
    for key in metrics.keys():
        mu = np.mean(metrics[key])
        std = np.std(metrics[key]) if len(metrics[key]) > 1 else 0.0
        mu_std_metrics[key] = "%.4f\u00B1%.4f" % (mu, std)

    return metrics, mu_std_metrics


def get_optimal_seq(query, seq_list):
    best_score = -1
    best_chain = None
    for key, seq in seq_list.items():
        seq = "".join(seq).strip()
        if len(seq) == 0:
            continue
        score = get_similarity(query, seq)
        if score > best_score:
            best_score = score
            best_chain = key

    if best_chain is None:
        raise RuntimeError(f"mismatch on sequence.")

    return best_chain


def extract_chain_from_pdb(chain_id, pdb_path):
    chain_id = list(chain_id)
    f_pdb = open(pdb_path)
    sc_pdb = f_pdb.read().splitlines()
    f_pdb.close()
    rs = []
    for line in sc_pdb:
        if line.startswith("ATOM") and line[21] in chain_id:
            rs.append(line)

    return rs


def merge_chain(pdb_file_path: str, save_pdb_path: str):
    with open(pdb_file_path, "r") as f:
        lines = []
        curr_num = 0
        last_chain = ""
        chain_name = "AB"
        chain_idex = -1
        last_num = -1
        for line in f.readlines():
            if line.startswith("ATOM"):
                num = int(line[22:26])
                chain = line[21]
                atom_name = line[12:17].strip()
                res3 = line[17:20].strip()
                if num != last_num:
                    curr_num += 1
                    last_num = num
                if chain != last_chain:
                    chain_idex += 1
                    last_chain = chain
                    curr_num = 1
                num_to_str = "%4d" % curr_num
                line = line[:21] + chain_name[chain_idex] + num_to_str + line[26:]
            lines.append(line)
    with open(save_pdb_path, "w") as f:
        for line in lines:
            f.write(line)


def compute_scores(merged_tmp_pdb1_path, merged_tmp_pdb2_path, out=False):
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DockQ.py")
    if not os.path.exists(script_path):
        raise ValueError("DockQ script not exist")
    cmd = (
        "python "
        + script_path
        + " "
        + merged_tmp_pdb2_path
        + " "
        + merged_tmp_pdb1_path
    )

    score = 0.0
    try:
        with os.popen(cmd, "r") as p: 
            score_result = p.read()
            score = float(score_result.strip().splitlines()[-1][5:])
            LRMSD = float(score_result.strip().splitlines()[-2][4:])
            iRMSD = float(score_result.strip().splitlines()[-3][4:])
            Fnonnat = float(score_result.strip().splitlines()[-4].split()[1])
            Fnat = float(score_result.strip().splitlines()[-5].split()[1]) 
    except Exception as e:
        score = -1
        logger.warning("DockQ failed")
    return score, (LRMSD, iRMSD, Fnonnat, Fnat)


def get_residure_atom(pdb_str):
    last_num = -1
    rs = []
    res_atom = []
    for line in pdb_str:
        if line.startswith("ATOM"):
            num = int(line[22:26])
            if num != last_num:
                last_num = num
                if len(res_atom) != 0:
                    rs.append(res_atom)
                res_atom = []
            res_atom.append(line)
    return rs


def get_atom_type(lines):
    rs = []
    for line in lines:
        rs.append(line[12:17].strip())
    return rs


def filter_atom(lines, type):
    rs = []
    for line in lines:
        if line[12:17].strip() in type and line[26].strip() == "":
            rs.append(line)
    return rs


def find_same_atom(res1_atom, res2_atom):
    res1_atom_type = get_atom_type(res1_atom)
    res2_atom_type = get_atom_type(res2_atom)
    common_atom = list(set(res1_atom_type) & set(res2_atom_type))
    return filter_atom(res1_atom, common_atom), filter_atom(res2_atom, common_atom)


def remove_unpair_atom(atom1, atom2):
    res_atom_1 = get_residure_atom(atom1)
    res_atom_2 = get_residure_atom(atom2)
    i = 0
    j = 0
    rs1_line = []
    rs2_line = []
    while i < len(res_atom_1) and j < len(res_atom_2):
        res_name_1 = res_atom_1[i][0][17:20].strip()
        res_name_2 = res_atom_2[j][0][17:20].strip()
        if res_name_1 != res_name_2:
            if len(res_atom_1) > len(res_atom_2):
                i += 1
            elif len(res_atom_1) == len(res_atom_2):
                i += 1
                j += 1
            else:
                j += 1
        else:
            rs1, rs2 = find_same_atom(res_atom_1[i], res_atom_2[j])
            rs1_line.extend(rs1)
            rs2_line.extend(rs2)
            i += 1
            j += 1

    return rs1_line, rs2_line


def compute_multi_chain_score(pdb1, pdb2, max_dockq=False):

    _, seqs1, _ = parse_pdb(pdb1)
    _, seqs2, _ = parse_pdb(pdb2)
    if len(seqs1.keys()) < 2:
        raise RuntimeError("DockQ scores take at least two chains as input.")

    if len(seqs1.keys()) != len(seqs2.keys()):
        raise RuntimeError(f"the compared pdb={pdb1} does not match with {pdb2}.")
    
    with tmpdir_manager(base_dir="/tmp") as pdb_tmp_dir:
        pdb1_chain_id = []
        pdb2_chain_id = []
        reordered_seqs2 = dict()
        for chain_id, seq in seqs1.items():
            query = "".join(seq).strip()
            matched_chain_id = get_optimal_seq(query, seqs2)
            pdb1_chain_id.append(chain_id)
            pdb2_chain_id.append(matched_chain_id)
            reordered_seqs2[matched_chain_id] = seqs2[matched_chain_id]
            seqs2.pop(matched_chain_id)
       
        idx = [i for i in range(len(pdb1_chain_id))]
        ret = dict()
        ret_ffil = dict()
        if max_dockq:
            maxDockQ = 0
        for item in list(itertools.combinations(idx, 2)):
            tmp_pdb1_path = os.path.join(pdb_tmp_dir, f"pdb{item[0]}_{item[1]}_1.pdb")
            tmp_pdb2_path = os.path.join(pdb_tmp_dir, f"pdb{item[0]}_{item[1]}_2.pdb")
            merged_tmp_pdb1_path = os.path.join(
                pdb_tmp_dir, f"pdb1_{item[0]}_{item[1]}_merged.pdb"
            )
            merged_tmp_pdb2_path = os.path.join(
                pdb_tmp_dir, f"pdb2_{item[0]}_{item[1]}_merged.pdb"
            )
            
            for chain_idx in item:
                atom1_list = extract_chain_from_pdb(pdb1_chain_id[chain_idx], pdb1)
                atom2_list = extract_chain_from_pdb(pdb2_chain_id[chain_idx], pdb2)

                atom1_list, atom2_list = remove_unpair_atom(atom1_list, atom2_list)
                with open(tmp_pdb1_path, "a") as f:
                    for line in atom1_list:
                        f.write(line + "\n")
                with open(tmp_pdb2_path, "a") as f:
                    for line in atom2_list:
                        f.write(line + "\n")
            merge_chain(tmp_pdb1_path, merged_tmp_pdb1_path)
            merge_chain(tmp_pdb2_path, merged_tmp_pdb2_path)

            score, ffil = compute_scores(merged_tmp_pdb1_path, merged_tmp_pdb2_path)
            
            if score == -1:
                continue
            ret[f"{pdb1_chain_id[item[0]]}_{pdb1_chain_id[item[1]]}"] = score
            ret_ffil[f"{pdb1_chain_id[item[0]]}_{pdb1_chain_id[item[1]]}"] = ffil
            if max_dockq:
                is1, _ = is_antibody("".join(seqs1[pdb1_chain_id[item[0]]]).strip())
                is2, _ = is_antibody("".join(seqs1[pdb1_chain_id[item[1]]]).strip())
                if (is1 == 0 and is2 == 1) or (is1 == 1 and is2 == 0):
                    
                    ret_ffil[f"{pdb1_chain_id[item[0]]}_{pdb1_chain_id[item[1]]}"]=(ffil,1)
                    if score > maxDockQ:
                        maxDockQ = score
                else:
                    flag = (0)
                    ret_ffil[f"{pdb1_chain_id[item[0]]}_{pdb1_chain_id[item[1]]}"]=(ffil,0)  

                        
    if max_dockq:
        return ret, maxDockQ, ret_ffil
    else:
        return ret, None, ret_ffil


if __name__ == "__main__":
    args = get_args()

    pdb_list_1 = get_fasta_file_list(args.pdb_paths_1)
    pdb_list_2 = get_fasta_file_list(args.pdb_paths_2)
    # pdb_list_1 = args.pdb_paths_1
    # pdb_list_2 = args.pdb_paths_2
    results = list()

    if args.max_dockq:
        maxQs = []
    cnt_success = 0
    pdb_names, df, model_list = [], [], []
    #pdb_list_1=pdb_list_1[:7]  ##only test

    for i, pdb_path in enumerate(pdb_list_1):
        pdb_name = pdb_path.split("/")[-1]
        smp_id = pdb_list_2[i].split('/')[-3:]
        smp_id='_'.join(smp_id)
        
        if not os.path.isfile(pdb_list_1[i]):
            logger.info(f"the compared pdb={pdb_list_1[i]} is NOT existed .")
            continue
        if not os.path.isfile(pdb_list_2[i]):
            logger.info(
                f"the compared pdb={pdb_name} is NOT existed in {pdb_list_2[i]}."
            )
            continue
        try:
            ret, maxDockQ, ret_ffil = compute_multi_chain_score(
                pdb_list_1[i], pdb_list_2[i], max_dockq=args.max_dockq
            )
        except Exception as e:
            logger.warning(e)
            logger.warning(f"catch unknow error, skip {pdb_name}")
            continue
        if args.max_dockq:
            logger.info("comparison %s: %s, maxDockQ: %.4f" % (pdb_name, ret, maxDockQ))
        else:
            logger.info("comparison %s: %s" % (pdb_name, ret))

        pdb_names.append(pdb_name)
        model_list.append(smp_id)  
        results.append(ret)
        if args.max_dockq:
            maxQs.append(maxDockQ)
        df.append([str(ret), maxDockQ, ret_ffil] if args.max_dockq else str(ret))
        cnt_success += 1
    _, mu_std_metrics = evaluation(results)
    logger.info(f"total cases: {len(pdb_list_1)}, success: {cnt_success}")
    logger.info(mu_std_metrics)
    
    if args.max_dockq:
        logger.info(
            f"maxDockQ: {np.array(maxQs).mean() if maxQs else 0}\u00B1{np.array(maxQs).std() if maxQs else 0}"
        )

    if args.output_file_path is not None:
        import pandas as pd

        # df = pd.DataFrame(
        #     df,
        #     columns=["dockq_results", "maxDockQ","ffil_results"]
        #     if args.max_dockq
        #     else ["dockq_results"],
        # )

        df = pd.DataFrame(df)
        df["pdb"] = smp_id
        df.to_csv(args.output_file_path, index=False,mode="a",header=None)
