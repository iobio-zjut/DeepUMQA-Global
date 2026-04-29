# DeepUMQA-Global
DeepUMQA-Global is a deep learning framework for estimating the fold accuracy of protein structural models at both the global complex (pScore) and interface (ipScore) levels. We introduce a structure–sequence cross-consistency mechanism that captures the bidirectional compatibility between the three-dimensional structure and the amino acid sequence.

## ⭐**Overall workflow for the DeepUMQA-Global**⭐
![DeepUMQA-Global pipeline](pipeline.png)

---

## 🚀 **Getting Started**

### 1.🛠**Download DeepAAAssembly package**

```
git clone --recursive https://github.com/iobio-zjut/DeepAAAssembly 
```

#### Quick Start

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

### 2.📥**🔧 Software Requirements**

To run this project, you need the following dependencies installed **(or use the provided Singularity container)**:

- Python ≥ 3.8  
- PyTorch 1.11.0  
- PyTorch Geometric 2.0.4  
- PyRosetta ≥ 2021.38+release.4d5a969
- **Voronota** ([GitHub](https://github.com/kliment-olechnovic/voronota?tab=MIT-1-ov-file)) | [MIT](https://opensource.org/license/mit)
  - Installed in a conda environment named `vorolf` 
- **Foldseek** ([GitHub](https://github.com/steineggerlab/foldseek)) | [GPL-3.0](https://opensource.org/licenses/GPL-3.0)  
  - Installed in a conda environment named `foldseek-multimer`  
- **ProteinMPNN** ([GitHub](https://github.com/dauparas/ProteinMPNN)) | [MIT](https://opensource.org/license/mit)  
  - Installed in a conda environment named `MPNN`
 
> Alternatively, you can run everything inside a [Singularity container](https://zenodo.org/api/records/19888061/draft/files/DeepUMQAGlobal.sif/content) to avoid dependency issues (container size: 6.64 GB).
 
### 2.📥**🔧 Data Preparation**

- **PDB100**([PDB100](https://steineggerlab.s3.amazonaws.com/foldseek/pdb100.tar.gz))
  - Template database for SAGS feature extraction (complex)
- **PDB_AFDB_207187**([PDB_AFDB_207187](http://zhanglab-bioinf.com/PAthreader/database/PDB_AFDB_207187.tar))
  - Template database for SAGS feature extraction (monomer)



---







---

### 📚 Resources

[CASP16 EMA Data](https://predictioncenter.org/download_area/CASP16/)
Includes predicted models, experimental structures, and EMA results.

[PDB-2024 Targets](https://www.rcsb.org/): Estimation for docking-based models.

---

## 📬 Contact (Supervisor)

**Prof. Guijun Zhang**  
College of Information Engineering  
Zhejiang University of Technology, Hangzhou 310023, China  
✉️ Email: [zgj@zjut.edu.cn](mailto:zgj@zjut.edu.cn)
