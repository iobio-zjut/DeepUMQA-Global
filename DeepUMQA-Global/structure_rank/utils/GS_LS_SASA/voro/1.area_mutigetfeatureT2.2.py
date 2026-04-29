import os
import multiprocessing
import sys
# # 读取包含子文件夹名的文件
# # 根目录
if len(sys.argv) != 4:
    print("Usage: python 1.area_mutigetfeatureT2.2.py subfolder_file root_folder")
    sys.exit(1)
subfolder_file = sys.argv[1]
root_folder = sys.argv[2]
result_folder = sys.argv[3]

i = 0

def process_pdb_file(pdb_file):
    global i
    if pdb_file.endswith(".pdb"):
        # 提取文件名作为注释，例如 m001.pdb 对应 annotation 为 m001
        annotation = os.path.splitext(pdb_file)[0]

        # 构建完整的文件路径
        pdb_file_path = os.path.join(root, pdb_file)
        
        # 创建目标文件夹在用户主目录下的新文件夹
        destination_folder = os.path.expanduser(result_folder + root[len(root_folder):])

        os.makedirs(destination_folder, exist_ok=True)
        
        #print(destination_folder,i)
        i += 1

        target_file_path = os.path.join(destination_folder, f'{annotation}.txt')

        # if os.path.exists(target_file_path):
        #     print(f"文件 {target_file_path} 已存在，跳过。")
        #     return

        # 第一个命令
        os.system(f"./voronota get-balls-from-atoms-file --annotated < {pdb_file_path} > {pdb_file_path}.txt")
        # 如果文件已经存在，则跳过执行下一个文件
        # if os.path.exists(target_file_path):
        #     print(f"文件 {target_file_path} 已存在，跳过。")
        #     os.remove(f"{pdb_file_path}.txt")
        #     return

        # 第二个命令
        os.system(f"./voronota calculate-contacts --annotated < {pdb_file_path}.txt > {target_file_path}")

        # 删除中间文件
        os.remove(f"{pdb_file_path}.txt")

# 读取包含子文件夹名的文件
with open(subfolder_file, 'r') as f:
    subfolders = f.read().splitlines()

# 遍历每个子文件夹
for subfolder in subfolders:
    # 构建子文件夹的完整路径
    subfolder_path = os.path.join(root_folder, subfolder)
    
    # 遍历子文件夹中的pdb文件
    for root, dirs, files in os.walk(subfolder_path):
        pool = multiprocessing.Pool(processes=8)  # 使用16个进程
        pool.map(process_pdb_file, files)
        pool.close()
        pool.join()

# # 删除已处理的子文件夹对应的行
# with open(subfolder_file, 'r') as f:
#     lines = f.readlines()

# with open(subfolder_file, 'w') as f:
#     for line in lines:
#         if line.strip() not in subfolders:
#             f.write(line)


