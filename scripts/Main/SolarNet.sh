. scripts/common_solar.sh

model_name=SolarNet

run_main_experiment () {
  channel_mode=$1
  data_path_name=$2
  model_id_name=$3
  solar_site_name=$4
  latitude=$5
  longitude=$6
  solar_timezone=$7
  enc_in=$8

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
        --channel_mode $channel_mode \
        --solar_site $solar_site_name \
        --latitude $latitude \
        --longitude $longitude \
        --solar_timezone $solar_timezone \
        --seq_len $SOLAR_SEQ_LEN \
        --pred_len $pred_len \
        --enc_in $enc_in \
        --cycle 288 \
        --train_epochs 30 \
        --patience 5 \
        --itr 1 --batch_size 64 --learning_rate 0.003 --random_seed $random_seed
  done
  done
}

for channel_mode in cd ci
do
  solar_sites | while read -r data_path_name model_id_name solar_site_name latitude longitude solar_timezone enc_in
  do
    run_main_experiment $channel_mode $data_path_name $model_id_name $solar_site_name $latitude $longitude $solar_timezone $enc_in
  done
done
