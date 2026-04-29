from typing import Optional
import tempfile
import contextlib
import os
import shutil
from rapidfuzz import distance

from structure_rank.tool.residue_constants import restype_3to1
from structure_rank.tool import protein

from structure_rank.tool.logger import Logger

logger = Logger.logger


@contextlib.contextmanager
def tmpdir_manager(base_dir: Optional[str] = None):
    """Context manager that deletes a temporary directory on exit."""
    tmpdir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def is_antibody(seq, scheme="imgt", ncpu=4):
    from anarci import anarci

    seqs = [("0", seq)]
    numbering, alignment_details, hit_tables = anarci(
        seqs, scheme=scheme, output=False, ncpu=ncpu
    )
    if numbering[0] is None:
        return False, None

    if numbering[0] is not None and len(numbering[0]) > 1:
        logger.warning("There are %d domains in %s" % (len(numbering[0]), seq))

    chain_type = alignment_details[0][0]["chain_type"].lower()
    if chain_type is None:
        return False, None
    else:
        return True, chain_type


def parse_pdb(input_pdb, pdb_chain_id_selected=None, use_chain_id_selected=False):
    # input: pdb_name
    # output:
    # seq_list : {A:[], B:[], C:[]}
    # index_list : [A,B,C]
    # atom_list : {A:{index1:{atom1:[], atom2:[]...}, index2:{}....},...}
    # pdb line:
    # ATOM    994  CE2 PHE A  64      11.485   1.004  -2.799  1.00 97.61           C
    # ATOM   1624 HE21 GLN A 105      -4.248  13.875  -3.152  1.00 97.99           H
    index_list = []

    seq_list = {}
    atom_list = {}

    last_index = ""
    input_pdb_file_name = os.path.basename(input_pdb)
    basename, ext = os.path.splitext(input_pdb_file_name)
    if (
        basename[:6] != "ranked"
        and len(basename.split("_")) > 1
        and use_chain_id_selected
    ):
        pdb_chain_id_selected = basename.split("_")[1]

    with open(input_pdb, "r") as f:
        pdb_str = f.read()

    protein_object = protein.from_pdb_string(
        pdb_str,
        chain_id=pdb_chain_id_selected,
        use_filter_atom=True,
        is_multimer=False,
        return_id2seq=False,
    )

    pdb_lines_from_pdb = protein.to_pdb(protein_object)
    pdb_lines_from_pdb = pdb_lines_from_pdb.split("\n")

    for line in open(input_pdb, "r"):
        if len(line) < 4:
            continue
        if line[0:4] != "ATOM":
            continue

        atom_name = line[12:17].strip()
        res3 = line[17:20].strip()
        if res3 not in restype_3to1.keys():
            continue
        chain = line[21]

        index = line[22:27].strip()
        coor_list = [line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]
        coor_list_float = []
        for item in coor_list:
            if item != "":
                try:
                    coor_list_float.append(float(item))
                except:
                    raise RuntimeError(
                        f"Found an unknown coordinate:{coor} in {line} of {input_pdb}!"
                    )

        if len(coor_list_float) != 3:
            raise RuntimeError(
                f"Found an unknown coordinate:{coor} in {line} of {input_pdb}!"
            )

        if chain not in index_list:
            index_list.append(chain)
            seq_list[chain] = []
            atom_list[chain] = {}

        res1 = restype_3to1[res3]
        if last_index != index:
            seq_list[chain].append(res1)
            last_index = index

        if index not in atom_list[chain].keys():
            atom_list[chain][index] = {}

        atom_list[chain][index][atom_name] = coor_list_float

    return index_list, seq_list, atom_list


def get_similarity(str1, str2):
    """
    Args:
        str1: query sequence
        str2: input sequence
    Returns: similarity score between 0 and 1.
    """

    return distance.Levenshtein.similarity(str1, str2)
