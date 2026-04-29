import os
from tqdm import tqdm
import sys
path = sys.argv[1]
output = sys.argv[2]
ckpt_list = os.listdir(path)
with open(output,'w') as f:
    for ckpt in tqdm(ckpt_list):
        ckpt = ckpt.strip()
        ckpt_path = os.path.join(path,ckpt)
        f.write(ckpt_path+"\n")