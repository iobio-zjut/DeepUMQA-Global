# DeepUMQA-Global

DeepUMQA-Global is a single-model protein structure quality assessment pipeline. For each input model, the program outputs two types of scores:

* `global_score.csv`
* `interface_score.csv`

The current open-source version has been refactored into a standalone local pipeline that does not depend on SSH, SCP, remote node switching, or plaintext credentials. The only official entry point is `run_dual_inference.py` located in the root directory.

---

## Directory Structure

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

---

## Requirements

Minimum Python dependencies include:

* Python 3.8+
* PyTorch
* PyTorch Geometric
* NumPy
* SciPy
* pandas
* tqdm
* Biopython

External tools that may be required during feature generation:

* Foldseek
* Voronota / Voronota-normal
* ProteinMPNN
* PyRosetta

These tools are no longer hardcoded to specific development machine paths. You can provide them via:

* Adding them to your `PATH`
* Passing paths through command-line arguments
* Setting environment variables

---

## Checkpoints

A usable pretrained checkpoint is included:

```text
./checkpoints/val_best-epoch=26-val_loss=0.00057.ckpt
```

You can also specify a custom checkpoint file or directory:

```bash
--ckpt-path /path/to/checkpoints
```

If `--ckpt-path` points to a directory, the program will automatically scan for `*.ckpt`, `*.pt`, and `*.pth` files.

---

## Example Directory

```text
example/pdb      Example input models
example/query    Corresponding target query/reference structures
example/feature  Feature cache directory
example/output   Final output directory
```

A complete reusable example dataset is provided. When running the example:

* Existing features will be reused if available
* If `example/feature` is cleared, a full feature generation environment is required locally

---

## Quick Start

Run directly:

```bash
bash bin/run_pipeline.sh
```

This script uses the following defaults:

* `./example/pdb`
* `./example/query`
* `./example/feature`
* `./example/output`
* `./checkpoints`

If your default `python` is not the desired environment:

```bash
DEEPUMQA_PYTHON_BIN=/path/to/python bash bin/run_pipeline.sh
```

---

## Main Entry Usage

The official entry point is:

```bash
python run_dual_inference.py \
  --pdb-root ./example/pdb \
  --query-root ./example/query \
  --feature-root ./example/feature \
  --output-root ./example/output \
  --ckpt-path ./checkpoints
```

To explicitly specify Python environments:

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

---

## Output

After successful execution, only two result files are retained in `output-root`:

* `global_score.csv`
* `interface_score.csv`

Intermediate run directories are stored at:

```text
<feature-root>/.runs/<run_id>
```

Unless `--keep-temp` is specified, temporary files will be automatically cleaned up after execution.

---

## Common Parameters

### Path Parameters

* `--pdb-root`
* `--query-root`
* `--feature-root`
* `--output-root`
* `--ckpt-path`
* `--pdb-list`
* `--ckpt-list`

### Tool Path Parameters

* `--python-bin`
* `--foldseek-bin`
* `--mpnn-python`
* `--voro-python`
* `--pyrosetta-python`
* `--voro-exe-dir`
* `--sp-template-db`
* `--sp-monomer-template-db`
* `--afdb-dir`

### Runtime Control Parameters

* `--force`
* `--keep-temp`
* `--feature-workers`
* `--interface-workers`
* `--infer-workers`
* `--cb-cutoff`
* `--skip-feature-generation`
* `--gpu-max-length`
* `--max-length`

---

## Optional Environment Variables

* `DEEPUMQA_PYTHON_BIN`
* `DEEPUMQA_FOLDSEEK_BIN`
* `DEEPUMQA_MPNN_PYTHON`
* `DEEPUMQA_VORO_PYTHON`
* `DEEPUMQA_PYROSETTA_PYTHON`
* `DEEPUMQA_VORO_EXE_DIR`
* `DEEPUMQA_SP_TEMPLATE_DB`
* `DEEPUMQA_SP_MONOMER_TEMPLATE_DB`
* `DEEPUMQA_AFDB_DIR`

The `bin/run_pipeline.sh` script also supports overriding example paths:

* `DEEPUMQA_PDB_ROOT`
* `DEEPUMQA_QUERY_ROOT`
* `DEEPUMQA_FEATURE_ROOT`
* `DEEPUMQA_OUTPUT_ROOT`
* `DEEPUMQA_CKPT_PATH`

---

## Standalone Execution

The default pipeline runs entirely on a single machine. The open-source version does **not** require:

* SSH access to remote nodes
* SCP-based file transfer
* Remote host switching
* Plaintext credentials

If you plan to extend the pipeline to support distributed or remote execution, maintain that separately and do not reintroduce remote dependencies into the default workflow.

---
