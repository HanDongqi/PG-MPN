export CUDA_VISIBLE_DEVICES=1

python3 train_gnn.py \
  --task_folder data/NELL-betae \
  --output_dir log/nell/pgmpn \
  --checkpoint_path pretrain/cqd/NELL-model-rank-1000-epoch-100-1602499096.pt \
  --agg_func mean \
  --epoch 100 \
  --reasoner pgmpn \
  --embedding_dim 1000 \
  --hidden_dim 8192 \
  --temp 0.1 \
  --batch_size_eval_dataloader 8 \
  --batch_size_eval_truth_value 1 \
  --device cuda:0 \
  --batch_size 512 \
  --learning_rate 5e-5 \
  --num_layers 1
