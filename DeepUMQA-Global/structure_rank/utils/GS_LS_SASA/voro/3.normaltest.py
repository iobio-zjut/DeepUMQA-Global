import os
import numpy as np
import re
from collections import defaultdict
import sys

def extract_normal(contact_file, npz_file):
    try:
        residue_pairs_normals = defaultdict(list)
        chain1_number = {}
        current_index = 1
        
        # 合并两次文件遍历为一次
        with open(contact_file, 'r') as input_f:
            for line in input_f:
                if line.startswith('c<'):
                    parts = line.strip().split()
                    if parts[1] == 'c<solvent>' or parts[0] == 'c<solvent>':
                        continue
                    
                    # 提取链名和残基信息
                    chainname1 = re.search(r'c<(\w+)>r<(\d+)>', parts[0]).group(1)
                    chainname2 = re.search(r'c<(\w+)>r<(\d+)>', parts[1]).group(1)
                    residue1 = re.search(r'c<(\w+)>r<(\d+)>', parts[0]).group(2)
                    residue2 = re.search(r'c<(\w+)>r<(\d+)>', parts[1]).group(2)
                    chain_residue1 = f"{chainname1} {residue1}"
                    chain_residue2 = f"{chainname2} {residue2}"
                    
                    # 为未见过的残基分配编号
                    if chain_residue1 not in chain1_number:
                        chain1_number[chain_residue1] = current_index
                        current_index += 1
                    if chain_residue2 not in chain1_number:
                        chain1_number[chain_residue2] = current_index
                        current_index += 1
                    
                    if chain_residue1 == chain_residue2:
                        continue
                    
                    # 提取法向量
                    normal = [float(x) for x in parts[4][1:-1].split(',')]
                    
                    # 使用排序后的残基编号作为键
                    residue_pair = tuple(sorted([
                        chain1_number[chain_residue1], 
                        chain1_number[chain_residue2]
                    ]))
                    
                    # 将法向量加入到字典中
                    residue_pairs_normals[residue_pair].append(normal)
        
        # 确定张量大小
        if not chain1_number:
            print(f"警告: 文件 {contact_file} 不包含有效数据")
            return
        
        max_residue_number = max(chain1_number.values())
        tensor_normal = np.zeros((max_residue_number, max_residue_number, 3))
        
        # 填充张量
        for residue_pair, normal_list in residue_pairs_normals.items():
            merged_normal = np.sum(normal_list, axis=0)
            magnitude = np.linalg.norm(merged_normal)
            
            # 处理零向量情况
            if magnitude == 0:
                normalized_normal = np.zeros(3)
            else:
                normalized_normal = merged_normal / magnitude
            
            residue_1, residue_2 = residue_pair
            # 调整为0-based索引
            tensor_normal[residue_1-1, residue_2-1] = normalized_normal
            tensor_normal[residue_2-1, residue_1-1] = normalized_normal
        
        # 保存为压缩的npz文件
        np.savez_compressed(npz_file, normal=tensor_normal.astype(np.float16))
        print(f"成功处理文件 {contact_file}，保存到 {npz_file}")
        
    except Exception as e:
        print(f"处理文件 {contact_file} 时发生错误: {e}")
        # 可以选择在这里记录更详细的错误日志

def process_folder(folder_path):
    folder_name = os.path.basename(folder_path)
    output_folder_path = os.path.join(result_folder, folder_name)
    
    os.makedirs(output_folder_path, exist_ok=True)
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            input_file = os.path.join(folder_path, file_name)
            # 检查文件大小是否不为0
            if os.path.getsize(input_file) != 0:
                npz_file = os.path.join(output_folder_path, f"{os.path.splitext(file_name)[0]}.npz")
                print("Processing file:", file_name)
                extract_normal(input_file, npz_file)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python 2.normal.py subfolder_file root_folder result_folder")
        sys.exit(1)

    subfolder_file = sys.argv[1]
    root_folder = sys.argv[2]
    result_folder = sys.argv[3]

    # 读取包含子文件夹路径的列表文件
    with open(subfolder_file, 'r') as f:
        for folder_path in f:
            folder_path = root_folder + "/" + folder_path.strip()
            process_folder(folder_path)