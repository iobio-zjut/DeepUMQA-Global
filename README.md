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
 
**Download [Singularity container](https://zenodo.org/api/records/19888061/draft/files/DeepUMQAGlobal.sif/content) (container size: 6.64 GB).**
 
### **📥 Data Preparation**

- **PDB100**([PDB100](https://steineggerlab.s3.amazonaws.com/foldseek/pdb100.tar.gz))
  - Template database for SAGS feature extraction (complex)
- **PDB_AFDB_207187**([PDB_AFDB_207187](http://zhanglab-bioinf.com/PAthreader/database/PDB_AFDB_207187.tar))
  - Template database for SAGS feature extraction (monomer)
- **PDB_AFDB_db** (Foldseek-formatted database converted from PDB_AFDB_207187)  
  Build the Foldseek database using:
  ```bash
  foldseek createdb PDB_AFDB_207187 PDB_AFDB_db

---

## 🏃 Running the Pipeline
### 🛠**Download DeepUMQA-Global package**

```
git clone --recursive https://github.com/iobio-zjut/DeepUMQA-Global 
```

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
---
The main pipeline runs inside a Singularity container.  
All required binaries and environments are pre-installed in the container.

### Directory Structure

```text
DeepUMQA-G_github/
├── run_dual_inference.py
├── bin/
│   └── run_DeepUMQAGlobal.sh
│   └── run_quickstart.sh
├── checkpoints/
│   └── best.ckpt
├── structure_rank/
├── example/
│   ├── pdb/
│   ├── query/
│   ├── feature/
│   └── output/
```

---

### 🧱 Environment Configuration

The following variables define the runtime environment and paths inside/outside the container:

| Variable | Description |
|----------|------------|
| `SCRIPT_DIR` | Directory of the pipeline script |
| `LOCAL_PROJECT_DIR` | Local project directory on the host machine. **Must be mounted into the container** |
| `CONTAINER_PROJECT_MOUNT` | Mount point inside container (`/repo`) |
| `PYTHON_BIN_IN_CONTAINER` | Python executable inside container |
| `FOLDSEEK_BIN_IN_CONTAINER` | Foldseek binary inside container (multimer version) |
| `VORO_EXE_DIR_IN_CONTAINER` | Voronota executable directory inside container |

---

### 📚 External Databases (Required)

These paths point to pre-downloaded template and reference databases:

| Variable | Description |
|----------|------------|
| `DEFAULT_SP_TEMPLATE_DB` | Path of Foldseek PDB100 database |
| `DEFAULT_SP_MONOMER_TEMPLATE_DB` | Path of PDB_AFDB_207187 |
| `DEFAULT_AFDB_DIR` | Path of PDB_AFDB_db |

---

### ⚠️ Important Notes

- `LOCAL_PROJECT_DIR` must be correctly set and mounted into the container at runtime.
- All other paths inside the container should **not be modified unless necessary**.
- The pipeline assumes Singularity execution by default.

---

### ⚡ Quick Start

Run the pipeline using default settings:

```bash
bash bin/run_pipeline.sh

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
