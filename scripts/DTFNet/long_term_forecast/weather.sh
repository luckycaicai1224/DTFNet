seq_len=336
model_name=DTFNet

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
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
      --enc_in 21 \
      --e_layers 2 \
      --n_branches 2 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
      --head_dropout 0.1 \
      --patch_len_ls '24, 96' \
      --stride_ls '12, 48' \
      --des 'Exp' \
      --lradj 'TST' \
      --padding_patch 'end' \
      --rel_pe 'rel_sin' \
      --train_epochs 100 \
      --patience 10 \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 \
      --tfactor 3 \
      --wavelet_layers 5 \
      --wavelet_type 'db5' \
      --wavelet_mode 'periodization' \
      --wavelet_dim 64 \
      --initTLV 1.0 \
      --dilation_rates '1, 2'
done
