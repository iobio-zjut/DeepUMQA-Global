# DeepUMQA-Global

DeepUMQA-Global 是一个单模型蛋白结构质量评估流程。对每个输入模型，程序最终输出两类分数：

- `global_score.csv`
- `interface_score.csv`

当前开源版本已经整理为单机版流程，默认不依赖 SSH、SCP、远程节点切换或明文凭据。正式唯一主入口是根目录下的 `run_dual_inference.py`。

## 目录结构

```text
DeepUMQA-G_github/
├── run_dual_inference.py
├── bin/
│   └── run_pipeline.sh
├── checkpoints/
│   ├── README.md
│   └── val_best-epoch=26-val_loss=0.00057.ckpt
├── structure_rank/
├── example/
│   ├── pdb/
│   ├── query/
│   ├── feature/
│   └── output/
├── requirements.txt
└── README.md
```

## 运行要求

Python 依赖至少包括：

- Python 3.8+
- PyTorch
- PyTorch Geometric
- NumPy
- SciPy
- pandas
- tqdm
- Biopython

特征生成阶段可能用到的外部工具包括：

- Foldseek
- Voronota / Voronota-normal
- ProteinMPNN
- PyRosetta

这些工具不再写死在开发机路径里。你可以通过以下方式提供：

- 放到 `PATH`
- 通过命令行参数传入
- 通过环境变量传入

## checkpoints

仓库内已保留当前示例可用的训练权重：

```text
./checkpoints/val_best-epoch=26-val_loss=0.00057.ckpt
```

也可以通过下面参数改用其他权重文件或目录：

```bash
--ckpt-path /path/to/checkpoints
```

如果 `--ckpt-path` 指向目录，程序会自动扫描其中的 `*.ckpt`、`*.pt`、`*.pth`。

## example 目录说明

```text
example/pdb      示例输入模型
example/query    对应 target 的 query/reference 模型
example/feature  示例特征与运行缓存目录
example/output   最终输出目录
```

仓库内已经放好一套可复用的 `example` 数据。直接跑示例时，程序会优先复用已有特征；如果你清空 `example/feature`，则需要本机具备完整特征生成环境。

## 最快使用方式

直接运行：

```bash
bash bin/run_pipeline.sh
```

这个脚本默认就会使用：

- `./example/pdb`
- `./example/query`
- `./example/feature`
- `./example/output`
- `./checkpoints`

如果默认 `python` 不是你想用的环境，可以这样运行：

```bash
DEEPUMQA_PYTHON_BIN=/path/to/python bash bin/run_pipeline.sh
```

## 主入口用法

正式主入口仍然是：

```bash
python run_dual_inference.py \
  --pdb-root ./example/pdb \
  --query-root ./example/query \
  --feature-root ./example/feature \
  --output-root ./example/output \
  --ckpt-path ./checkpoints
```

如果你要显式指定环境里的 Python，也可以这样：

```bash
python run_dual_inference.py \
  --pdb-root ./example/pdb \
  --query-root ./example/query \
  --feature-root ./example/feature \
  --output-root ./example/output \
  --ckpt-path ./checkpoints \
  --python-bin /path/to/python \
  --mpnn-python /path/to/python \
  --voro-python /path/to/python \
  --pyrosetta-python /path/to/python
```

## 输出结果

正常运行结束后，`output-root` 下只保留两个结果文件：

- `global_score.csv`
- `interface_score.csv`

中间运行目录默认写到：

```text
<feature-root>/.runs/<run_id>
```

如果不加 `--keep-temp`，运行结束后会自动清理临时目录。

## 常用参数

路径参数：

- `--pdb-root`
- `--query-root`
- `--feature-root`
- `--output-root`
- `--ckpt-path`
- `--pdb-list`
- `--ckpt-list`

工具路径参数：

- `--python-bin`
- `--foldseek-bin`
- `--mpnn-python`
- `--voro-python`
- `--pyrosetta-python`
- `--voro-exe-dir`
- `--sp-template-db`
- `--sp-monomer-template-db`
- `--afdb-dir`

运行控制参数：

- `--force`
- `--keep-temp`
- `--feature-workers`
- `--interface-workers`
- `--infer-workers`
- `--cb-cutoff`
- `--skip-feature-generation`
- `--gpu-max-length`
- `--max-length`

## 可选环境变量

- `DEEPUMQA_PYTHON_BIN`
- `DEEPUMQA_FOLDSEEK_BIN`
- `DEEPUMQA_MPNN_PYTHON`
- `DEEPUMQA_VORO_PYTHON`
- `DEEPUMQA_PYROSETTA_PYTHON`
- `DEEPUMQA_VORO_EXE_DIR`
- `DEEPUMQA_SP_TEMPLATE_DB`
- `DEEPUMQA_SP_MONOMER_TEMPLATE_DB`
- `DEEPUMQA_AFDB_DIR`

`bin/run_pipeline.sh` 还支持这些示例目录覆盖环境变量：

- `DEEPUMQA_PDB_ROOT`
- `DEEPUMQA_QUERY_ROOT`
- `DEEPUMQA_FEATURE_ROOT`
- `DEEPUMQA_OUTPUT_ROOT`
- `DEEPUMQA_CKPT_PATH`

## 单机说明

默认流程全部在当前机器执行。开源版主流程不再要求：

- SSH 到其他节点
- SCP 分发文件
- 远程 host 跳转
- 明文口令

如果你后续要自行扩展远程模式，请单独维护，不要把远程依赖放回默认主流程。
