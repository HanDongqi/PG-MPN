# Conditional Logical Message Passing Transformer for Complex Query Answering #

---

This repository contains the official implementation for our KDD 2024 Research Track paper, [**Conditional Logical Message Passing Transformer for Complex Query Answering**](https://arxiv.org/abs/2402.12954):


```bibtex
@article{zhang2024conditional,
  title={Conditional Logical Message Passing Transformer for Complex Query Answering},
  author={Zhang, Chongzhi and Peng, Zhiping and Zheng, Junhao and Ma, Qianli},
  journal={arXiv preprint arXiv:2402.12954},
  year={2024}
}
```

In this work, we present Conditional Logical Message Passing Transformer (CLMPT), a special graph neural network for complex query answering over knowledge graphs. 
Extensive experiments results on several benchmark datasets show that CLMPT achieves a strong performance through a conditional message passing mechanism and a transformer-based node embedding update scheme.

Our code is based on an implementation of LMPNN available [here](https://github.com/HKUST-KnowComp/LMPNN).

Please follow the instructions next to reproduce the results in our experiments.


## 1. Prepare datasets and checkpoints ##

Specifically, one can run:

```bash
sh script_prepare.sh
```

Then, the datasets will be automatically downloaded in the `./data` folder and converted to the specific format. 
Also, the checkpoints for the pre-trained neural link predictor released by [CQD](https://github.com/uclnlp/cqd) will be properly loaded in the `./pretrain` folder. 


## 2. Train CLMPT and variants ##

Our default settings for CLMPT on FB15k-237: 

```bash
python3 train_gnn.py \
  --task_folder data/FB15k-237-betae \
  --output_dir log/FB15k-237/clmpt_2_8192_512b_5e-5_mean \
  --checkpoint_path pretrain/cqd/FB15k-237-model-rank-1000-epoch-100-1602508358.pt \
  --agg_func mean \
  --reasoner clmpt \
  --embedding_dim 1000 \
  --hidden_dim 8192 \
  --device cuda:0 \
  --batch_size 512 \
  --learning_rate 5e-5 \
  --num_layers 2
```

Our default settings for CLMPT on NELL995:

```bash
python3 train_gnn.py \
  --task_folder data/NELL-betae \
  --output_dir log/NELL/clmpt_2_8192_512b_5e-5_0.1 \
  --checkpoint_path pretrain/cqd/NELL-model-rank-1000-epoch-100-1602499096.pt \
  --agg_func mean \
  --reasoner clmpt \
  --embedding_dim 1000 \
  --device cuda:1 \
  --hidden_dim 8192 \
  --temp 0.1 \
  --batch_size_eval_dataloader 8 \
  --batch_size_eval_truth_value 1 \
  --batch_size 512 \
  --learning_rate 5e-5 \
  --num_layers 2 \
  --pre_norm
```

Our default settings for CLMPT on FB15k: 

```bash
python3 train_gnn.py \
  --task_folder data/FB15k-betae \
  --output_dir log/FB15k/clmpt_2_8192_1024b_1e-4 \
  --checkpoint_path pretrain/cqd/FB15k-model-rank-1000-epoch-100-1602520745.pt \
  --agg_func mean \
  --reasoner clmpt \
  --embedding_dim 1000 \
  --device cuda:1 \
  --hidden_dim 8192 \
  --batch_size 1024 \
  --learning_rate 1e-4 \
  --num_layers 2 \
  --pre_norm
```

When considering not using conditional message passing, just change the reasoner to 'lmpt', namely: 

```bash
python3 train_gnn.py \
  --task_folder data/FB15k-237-betae \
  --output_dir log/FB15k-237/lmpt_2_8192_512b_5e-5_mean \
  --checkpoint_path pretrain/cqd/FB15k-237-model-rank-1000-epoch-100-1602508358.pt \
  --agg_func mean \
  --reasoner lmpt \
  --embedding_dim 1000 \
  --hidden_dim 8192 \
  --device cuda:0 \
  --batch_size 512 \
  --learning_rate 5e-5 \
  --num_layers 2
```

When considering freezing the pre-trained neural link predictor, just add --freeze_nlp, namely:

```bash
python3 train_gnn.py \
  --task_folder data/FB15k-237-betae \
  --output_dir log/FB15k-237/clmpt_2_8192_512b_5e-5_mean_freeze \
  --checkpoint_path pretrain/cqd/FB15k-237-model-rank-1000-epoch-100-1602508358.pt \
  --agg_func mean \
  --reasoner clmpt \
  --embedding_dim 1000 \
  --hidden_dim 8192 \
  --device cuda:0 \
  --batch_size 512 \
  --learning_rate 5e-5 \
  --num_layers 2 \
  --freeze_nlp
```

## 3. Summarize the results for evaluation.

We use script `./read_eval_from_log.py` to summarize the results from the log files to evaluate the trained model performance:

```bash
python3 read_eval_from_log.py --log_file log/FB15k-237/clmpt_2_8192_512b_5e-5_mean/output.log
```

