export CUDA_VISIBLE_DEVICES=1

python3 train_gnn.py \
  --task_folder data/FB15k-237-betae \
  --output_dir log/fb15k-237/pgmpn \
  --checkpoint_path pretrain/cqd/FB15k-237-model-rank-1000-epoch-100-1602508358.pt \
  --agg_func mean \
  --epoch 100 \
  --reasoner pgmpn \
  --embedding_dim 1000 \
  --hidden_dim 8192 \
  --device cuda:0 \
  --batch_size 512 \
  --learning_rate 5e-5 \
  --num_layers 1
