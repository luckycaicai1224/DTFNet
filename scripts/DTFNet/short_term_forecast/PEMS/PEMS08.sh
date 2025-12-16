seq_len=336
model_name=DTFNet

root_path_name=./dataset/
data_path_name=PEMS08.npz
model_id_name=PEMS08
data_name=PEMS

random_seed=2021
for pred_len in 12 24 48 96
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 170 \
      --e_layers 2 \
      --n_branches 2 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3 \
      --fc_dropout 0.3 \
      --head_dropout 0.1 \
      --patch_len_ls '16, 24' \
      --stride_ls '8, 12' \
      --des 'Exp' \
      --padding_patch 'end' \
      --rel_pe 'rel_sin' \
      --train_epochs 100 \
      --patience 10 \
      --itr 1 \
      --batch_size 16 \
      --learning_rate 0.00001 \
      --tfactor 3 \
      --wavelet_layers 2 \
      --wavelet_type 'db5' \
      --wavelet_mode 'periodization' \
      --wavelet_dim 64 \
      --initTLV 0.2 \
      --dilation_rates '1'
done