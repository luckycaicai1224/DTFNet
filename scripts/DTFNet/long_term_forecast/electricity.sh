seq_len=336
model_name=DTFNet

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

random_seed=2021
for pred_len in 96 192 336 720
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
      --enc_in 321 \
      --e_layers 1 \
      --n_branches 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
      --head_dropout 0.1 \
      --patch_len_ls '8, 16, 48' \
      --stride_ls '4, 8, 24' \
      --des 'Exp' \
      --padding_patch 'end' \
      --rel_pe 'rel_sin' \
      --train_epochs 100 \
      --patience 10 \
      --lradj 'TST' \
      --pct_start 0.2 \
      --itr 1 \
      --batch_size 256 \
      --learning_rate 0.0001 \
      --tfactor 4 \
      --wavelet_layers 5 \
      --wavelet_type 'db5' \
      --wavelet_mode 'periodization' \
      --wavelet_dim 64 \
      --initTLV 1.0 \
      --dilation_rates '1, 2, 4'
done
