model_name=PatchTST

root_path_name=./dataset/
data_path_name=alabama5min.csv
model_id_name=alabama5min
data_name=custom

seq_len=144
for pred_len in 24 48 96 144 288
do
for random_seed in 2024
do
    uv run python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 137 \
      --cycle 288 \
      --train_epochs 30 \
      --patience 5 \
      --use_multi_gpu \
      --use_revin 0 \
      --itr 1 --batch_size 64 --learning_rate 0.0005 --random_seed $random_seed
done
done


root_path_name=./dataset/
data_path_name=newyork5min.csv
model_id_name=newyork5min
data_name=custom

seq_len=144
for pred_len in 24 48 96 144 288
do
for random_seed in 2024
do
    uv run python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 129 \
      --cycle 288 \
      --train_epochs 30 \
      --patience 5 \
      --use_multi_gpu \
      --use_revin 0 \
      --itr 1 --batch_size 64 --learning_rate 0.0005 --random_seed $random_seed
done
done


root_path_name=./dataset/
data_path_name=ohio5min.csv
model_id_name=ohio5min
data_name=custom

seq_len=144
for pred_len in 24 48 96 144 288
do
for random_seed in 2024
do
    uv run python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 95 \
      --cycle 288 \
      --train_epochs 30 \
      --patience 5 \
      --use_multi_gpu \
      --use_revin 0 \
      --itr 1 --batch_size 64 --learning_rate 0.0005 --random_seed $random_seed
done
done