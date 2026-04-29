# DeepUMQA-Global
DeepUMQA-Global is a deep learning framework for estimating the fold accuracy of protein structural models at both the global complex (pScore) and interface (ipScore) levels. We introduce a structure–sequence cross-consistency mechanism that captures the bidirectional compatibility between the three-dimensional structure and the amino acid sequence.

## ⭐**Overall workflow for the DeepUMQA-Global**⭐
![DeepUMQA-Global pipeline](pipeline.png)

---

## 🚀 **Getting Started**

### **🔧 Software Requirements**

To run this project, you need the following dependencies installed **(or use the provided Singularity container)**:

- Python ≥ 3.8  
- PyTorch 1.11.0  
- PyTorch Geometric 2.0.4
- biopython==1.78
- numpy==1.18.5
- pandas==1.3.5
- scipy==1.7.3
- PyRosetta ≥ 2021.38+release.4d5a969
- **Voronota** ([GitHub](https://github.com/kliment-olechnovic/voronota?tab=MIT-1-ov-file)) | [MIT](https://opensource.org/license/mit)
  - Installed in a conda environment named `vorolf` 
- **Foldseek** ([GitHub](https://github.com/steineggerlab/foldseek)) | [GPL-3.0](https://opensource.org/licenses/GPL-3.0)  
  - Installed in a conda environment named `foldseek-multimer`  
- **ProteinMPNN** ([GitHub](https://github.com/dauparas/ProteinMPNN)) | [MIT](https://opensource.org/license/mit)  
  - Installed in a conda environment named `MPNN`
 
> Alternatively, you can run everything inside a [Singularity container](https://zenodo.org/api/records/19888061/draft/files/DeepUMQAGlobal.sif/content) to avoid dependency issues (container size: 6.64 GB).
 
### **📥 Data Preparation**

- **PDB100**([PDB100](https://steineggerlab.s3.amazonaws.com/foldseek/pdb100.tar.gz))
  - Template database for SAGS feature extraction (complex)
- **PDB_AFDB_207187**([PDB_AFDB_207187](http://zhanglab-bioinf.com/PAthreader/database/PDB_AFDB_207187.tar))
  - Template database for SAGS feature extraction (monomer)

---

## 🏃 Running the Pipeline
### 🛠**Download DeepUMQA-Global package**

```
git clone --recursive https://github.com/iobio-zjut/DeepUMQA-Global 
```
---

### Directory Structure

```text
DeepUMQA-G_github/
├── run_dual_inference.py
├── bin/
│   └── run_pipeline.sh
├── checkpoints/
│   └── val_best-epoch=26-val_loss=0.00057.ckpt
├── structure_rank/
├── example/
│   ├── pdb/
│   ├── query/
│   ├── feature/
│   └── output/
├── requirements.txt
```

---

### ⚡ Quick Start

Run directly:

```bash
bash bin/run_pipeline.sh
```

This script uses the following defaults:

* `./example/pdb`
* `./example/query`
* `./example/feature`  # have extracted
* `./example/output`
* `./checkpoints`

If your default `python` is not the desired environment:

```bash
DEEPUMQA_PYTHON_BIN=/path/to/python bash bin/run_pipeline.sh
```


## 📌 Command-Line Usage

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

If you download the Singularity container, you can run as:
```bash
bash run_pipeline.sh \
  --pdb-root ./example/pdb \
  --query-root ./example/query \
  --feature-root ./example/feature \
  --output-root ./example/output \
  --ckpt-path ./checkpoints
```
---

### Optional Environment Variables

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

### 📚 Resources

[CASP16 EMA Data](https://predictioncenter.org/download_area/CASP16/)
Includes predicted models, experimental structures, and EMA results.

[PDB-2024 Targets](https://www.rcsb.org/): Estimation for docking-based models.

[CoDNaS](http://ufq.unq.edu.ar/codnas/): Estimation for alternative conformational states proteins.

---

## 📬 Contact (Supervisor)

**Prof. Guijun Zhang**  
College of Information Engineering  
Zhejiang University of Technology, Hangzhou 310023, China  
✉️ Email: [zgj@zjut.edu.cn](mailto:zgj@zjut.edu.cn)
