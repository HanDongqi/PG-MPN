# Complex Query Answering on Knowledge Graphs with Griffin and Polarity-Weighted Message Passing

This repository contains the official PyTorch implementation of the paper **"Complex Query Answering on Knowledge Graphs with Griffin and Polarity-Weighted Message Passing"** (Accepted by *Knowledge and Information Systems (KAIS)*).

In this work, we propose **PG-MPN** (Polarity-Aware Griffin Message Passing Network), a novel and efficient framework designed to handle conditional logic, especially logical negation, in Complex Query Answering (CQA) over incomplete Knowledge Graphs. By introducing the **Anchored Residual Transformation** and a **Polarity-Weighted Message Passing** module, combined with the linear-complexity **Griffin** sequence processor, PG-MPN achieves state-of-the-art performance and superior computational efficiency.

## 📌 Citation
If you find our code or paper useful for your research, please kindly cite our work:

```bibtex
@article{han2026complex,
  title={Complex Query Answering on Knowledge Graphs with Griffin and Polarity-Weighted Message Passing},
  author={Han, Dongqi and Zhang, Yao and Li, Fanghao and Lu, Hu and Wu, Shengli},
  journal={Knowledge and Information Systems},
  year={2026},
  publisher={Springer}
}
```
*(Note: The BibTeX will be updated with volume/page numbers once officially published online.)*

## 🙏 Acknowledgements

This codebase is built upon the excellent work of previous researchers. We would like to express our sincere gratitude to:
* The [**CLMPT**](https://github.com/qianlima-lab/CLMPT/tree/master) repository, which provided the foundational message-passing framework and conditional processing logic for our implementation.
* The [**CQD**](https://github.com/uclnlp/cqd) repository for providing the pre-processed datasets and the pre-trained neural link predictor checkpoints used in our experiments.

## 🚀 Usage Instructions

### 1. Prepare Datasets and Checkpoints

We provide a script to automatically prepare all required data. Specifically, you can run:

```bash
sh script_prepare.sh
```

By running this script:
1. The benchmark datasets (FB15k, FB15k-237, NELL995) will be automatically downloaded into the `./data` folder and converted to the specific format required by the model.
2. The checkpoints for the pre-trained neural link predictor, originally released by [CQD](https://github.com/uclnlp/cqd), will be properly downloaded and loaded into the `./pretrain` folder.

### 2. Train PG-MPN

To train the PG-MPN model on different benchmark datasets, we provide separate shell scripts with default hyperparameters. You can easily start the training process by running the corresponding script:

For **FB15k-237**:
```bash
sh train_fb15k237.sh
```

For **FB15k**:
```bash
sh train_fb15k.sh
```

For **NELL995**:
```bash
sh train_nell.sh
```

### 3. Evaluate and Summarize Results

Once the training is complete, the evaluation logs will be saved. We provide a python script `./read_eval_from_log.py` to automatically parse the log files and summarize the model's performance metrics (e.g., MRR on different query structures).

For example, to evaluate the trained model on FB15k-237, run:

```bash
python3 read_eval_from_log.py --log_file log/FB15k-237/pgmpn/output.log
```
*(Please adjust the `--log_file` path according to your actual log directory structure).*
