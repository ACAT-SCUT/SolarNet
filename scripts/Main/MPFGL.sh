. scripts/common_solar.sh

model_name=MPFGL

run_main_experiment () {
  data_path_name=$1
  model_id_name=$2
  enc_in=$3

  for pred_len in $SOLAR_PRED_LENS
  do
  for random_seed in $SOLAR_RANDOM_SEEDS
  do
      uv run python -u run.py \
        --is_training 1 \
        --root_path $SOLAR_ROOT \
        --data_path $data_path_name \
        --model_id ${model_id_name}_${SOLAR_SEQ_LEN}_${pred_len} \
        --model $model_name \
        --data $SOLAR_DATA \
        --features M \
        --seq_len $SOLAR_SEQ_LEN \
        --pred_len $pred_len \
        --enc_in $enc_in \
        --cycle 288 \
        --use_revin 1 \
        --d_model 64 \
        --d_ff 128 \
        --n_heads 4 \
        --train_epochs 30 \
        --patience 5 \
        --itr 1 --batch_size 64 --learning_rate 0.001 --random_seed $random_seed
  done
  done
}

solar_sites | while read -r data_path_name model_id_name _ _ _ _ enc_in
do
  run_main_experiment $data_path_name $model_id_name $enc_in
done
