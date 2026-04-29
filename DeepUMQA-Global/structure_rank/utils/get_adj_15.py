import numpy as np
import os
from tqdm import tqdm
import math
import re
import multiprocessing

def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx**2 + dy**2 + dz**2)

def read_atoms(file, chain=".", model=1):
    pattern = re.compile(chain)
    current_model = model
    atoms = []
    for line in file:
        line = line.strip()
        if line.startswith("ATOM"):
            type = line[12:16].strip()
            chain = line[21:22]
            if type == "CA" and re.match(pattern, chain):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atoms.append((x, y, z))
    return atoms

def compute_contacts(atoms, threshold):
    contacts = []
    for i in range(len(atoms)-2):
        for j in range(i+2, len(atoms)):
            if dist(atoms[i], atoms[j]) < threshold:
                contacts.append((i, j))
    return contacts

def write_output(contacts, file):
    for c in contacts:
        file.write("\t".join(map(str, c))+"\n")

def pdb_to_cm(file, threshold, chain=".", model=1):
    atoms = read_atoms(file, chain, model)
    return compute_contacts(atoms, threshold)

def get_adj(file, outfile):
    list_all = []
    contacts = pdb_to_cm(open(file, "r"), 15)
    list_all.append(contacts)
    np.savez(outfile, adj=list_all)

def process_target(target, input_file_path, out_file_path):
    target_path = os.path.join(input_file_path, target)
    out_target = os.path.join(out_file_path, target)
    if not os.path.exists(out_target):
        os.mkdir(out_target)

    decoy_list = os.listdir(target_path)
    for decoy in tqdm(decoy_list, desc=f"Processing {target}"):
        decoy = decoy.strip()
        decoy_path = os.path.join(target_path, decoy)
        if decoy_path.endswith('.pdb'):
            deocy = decoy.split('.')
            out_name = deocy[0] + '.npz'
            output = os.path.join(out_target, out_name)
            get_adj(decoy_path, output)

if __name__ == '__main__':
    # root_path = '/nfs_beijing_ai/ziqiang_2023/zxf/cui_temp/DeepUMQA-Multimer-tm/test1.22'
    # input_file_path = os.path.join(root_path, 'interface_pdb')
    # out_file_path = os.path.join(root_path, 'extra_adj')
    input_file_path = '/nfs_beijing_ai/liudong_2023/CMQA/Dataset/dataset_version2/interface_pdb'
    out_file_path = '/nfs_beijing_ai/ziqiang_2023/zxf/dataset/complex_15_adj'
    if not os.path.exists(out_file_path):
        os.mkdir(out_file_path)

    target_list = os.listdir(input_file_path)
    pool = multiprocessing.Pool(1)

    for target in target_list:
        target = target.strip()
        pool.apply_async(process_target, (target, input_file_path, out_file_path))

    pool.close()
    pool.join()
