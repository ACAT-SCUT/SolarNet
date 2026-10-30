#!/bin/sh

set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

PYTHON=${BASELINE_PYTHON:-$PROJECT_ROOT/.venv/bin/python}
DATASET_DIR=${BASELINE_DATASET_DIR:-$PROJECT_ROOT/dataset}
RESULTS_ROOT=${BASELINE_RESULTS_DIR:-$PROJECT_ROOT/results/all_baselines}
CHECKPOINTS_ROOT=${BASELINE_CHECKPOINTS_DIR:-$PROJECT_ROOT/checkpoints/all_baselines}
RUNS_ROOT=${BASELINE_RUNS_DIR:-$PROJECT_ROOT/logs/baseline_runs}

SEQ_LEN=${BASELINE_SEQ_LEN:-144}
PRED_LENS=${BASELINE_PRED_LENS:-"24 48 96 144 288"}
RANDOM_SEED=${BASELINE_RANDOM_SEED:-2024}
TRAIN_EPOCHS=${BASELINE_TRAIN_EPOCHS:-30}
TRAIN_RATIO=${BASELINE_TRAIN_RATIO:-0.6}
PATIENCE=${BASELINE_PATIENCE:-5}
BATCH_SIZE=${BASELINE_BATCH_SIZE:-64}
BATCH_FALLBACKS=${BASELINE_BATCH_FALLBACKS:-"32 16"}
NUM_WORKERS=${BASELINE_NUM_WORKERS:-8}
DATASET_PATTERN=${BASELINE_DATASET_PATTERN:-*.csv}
WAIT_FOR_RUN_DIR=${BASELINE_WAIT_FOR_RUN_DIR:-}
RETRY_FAILED_FROM=${BASELINE_RETRY_FAILED_FROM:-}
UNIT_TAG=${BASELINE_UNIT_TAG:-}
SOLAR_LATITUDE=${BASELINE_SOLAR_LATITUDE:-}
SOLAR_LONGITUDE=${BASELINE_SOLAR_LONGITUDE:-}
SOLAR_TIMEZONE=${BASELINE_SOLAR_TIMEZONE:-}

MODELS=${BASELINE_MODELS:-"Transformer Autoformer iTransformer PatchTST DLinear SparseTSF CycleNet TQNet GRU LSTM MPFGL"}

case "$TRAIN_RATIO" in
    0.1) TRAIN_RATIO_TAG=_train10 ;;
    0.2) TRAIN_RATIO_TAG=_train20 ;;
    0.6) TRAIN_RATIO_TAG= ;;
    *) echo "Unsupported BASELINE_TRAIN_RATIO=$TRAIN_RATIO (use 0.1, 0.2, or 0.6)." >&2; exit 1 ;;
esac
TASK_RATIO_TAG=${TRAIN_RATIO_TAG#_train}
[ -n "$TASK_RATIO_TAG" ] || TASK_RATIO_TAG=base

usage() {
    echo "Usage: sh scripts/Main/run_all_baselines.sh {start|worker RUN_DIR GPU_ID|status [RUN_DIR]}"
}

cycle_for_dataset() {
    case "$1" in
        *15min.csv) echo 96 ;;
        *5min.csv) echo 288 ;;
        *1hour.csv) echo 24 ;;
        *)
            echo "Unsupported dataset frequency: $1" >&2
            return 1
            ;;
    esac
}

learning_rate_for_model() {
    case "$1" in
        Transformer|Autoformer|PatchTST|GRU|LSTM|SolarNetGRU|SolarNetPatchTST) echo 0.0005 ;;
        iTransformer|MPFGL|SegRNN|DUET|TimeBase) echo 0.001 ;;
        Linear|DLinear|SparseTSF|CycleNet|TQNet|SolarNet|SolarNetLinear|SolarNetDLinear) echo 0.003 ;;
        *)
            echo "Unsupported model: $1" >&2
            return 1
            ;;
    esac
}

model_extra_args() {
    case "$1" in
        SparseTSF) echo "--period_len 4 --use_revin 0" ;;
        MPFGL) echo "--use_revin 1 --d_model 64 --d_ff 128 --n_heads 4" ;;
        SegRNN) echo "--d_model 128 --dropout 0 --rnn_type gru --dec_way pmf --seg_len 12 --channel_id 0 --revin 1" ;;
        DUET) echo "--CI 1 --d_model 128 --d_ff 256 --hidden_size 256 --factor 3 --n_heads 4 --e_layers 2 --dropout 0.2 --fc_dropout 0.05 --moving_avg 25 --activation gelu --num_experts 4 --k 2 --noisy_gating 1" ;;
        TimeBase) echo "--period_len 288 --basis_num 6 --use_period_norm 1 --use_orthogonal 0 --orthogonal_weight 0.2 --individual 0" ;;
        SolarNet|SolarNetLinear|SolarNetGRU|SolarNetDLinear|SolarNetPatchTST) echo "--channel_mode cd --solar_cycle_mode full --model_type mlp" ;;
        *) echo "--use_revin 0" ;;
    esac
}

batch_candidates() {
    candidates=$BATCH_SIZE
    for candidate in $BATCH_FALLBACKS
    do
        case " $candidates " in
            *" $candidate "*) ;;
            *) candidates="$candidates $candidate" ;;
        esac
    done
    echo "$candidates"
}

wait_for_prior_run() {
    [ -n "$WAIT_FOR_RUN_DIR" ] || return 0
    echo "WAITING_FOR $WAIT_FOR_RUN_DIR" > "$1"
    while :
    do
        active=0
        for unit_file in "$WAIT_FOR_RUN_DIR"/unit_gpu*.txt
        do
            [ -f "$unit_file" ] || continue
            unit_name=$(sed -n '1p' "$unit_file")
            if [ "$(systemctl --user is-active "$unit_name" 2>/dev/null || true)" = "active" ]; then
                active=1
            fi
        done
        [ "$active" -eq 0 ] && return 0
        sleep 30
    done
}

append_task_to_queue() {
    run_dir=$1
    task_index=$2
    task_line=$3
    if [ $((task_index % 2)) -eq 0 ]; then
        echo "$task_line" >> "$run_dir/queue_gpu0.tsv"
    else
        echo "$task_line" >> "$run_dir/queue_gpu1.tsv"
    fi
}

build_failed_queues() {
    run_dir=$1
    failed_tasks=$run_dir/failed_tasks.txt
    : > "$failed_tasks"
    for status_file in "$RETRY_FAILED_FROM"/status_gpu*.tsv
    do
        [ -f "$status_file" ] || continue
        awk -F '\t' '$1 == "FAILED" { print $3 }' "$status_file" >> "$failed_tasks"
    done
    sort -u -o "$failed_tasks" "$failed_tasks"

    task_index=0
    for source_queue in "$RETRY_FAILED_FROM"/queue_gpu*.tsv
    do
        [ -f "$source_queue" ] || continue
        while IFS='|' read -r model data_name data_stem enc_in cycle pred_len learning_rate
        do
            [ -n "$model" ] || continue
            task_name=${model}_${data_stem}_sl${SEQ_LEN}_pl${pred_len}_cycle${cycle}
            if grep -Fxq "$task_name" "$failed_tasks"; then
                task_line="$model|$data_name|$data_stem|$enc_in|$cycle|$pred_len|$learning_rate"
                append_task_to_queue "$run_dir" "$task_index" "$task_line"
                task_index=$((task_index + 1))
            fi
        done < "$source_queue"
    done
    echo "$task_index" > "$run_dir/total_tasks"
}

build_queues() {
    run_dir=$1
    queue0=$run_dir/queue_gpu0.tsv
    queue1=$run_dir/queue_gpu1.tsv
    : > "$queue0"
    : > "$queue1"

    if [ -n "$RETRY_FAILED_FROM" ]; then
        build_failed_queues "$run_dir"
        return
    fi

    task_index=0
    for data_file in "$DATASET_DIR"/$DATASET_PATTERN
    do
        [ -f "$data_file" ] || continue
        data_name=$(basename "$data_file")
        data_stem=${data_name%.csv}
        cycle=$(cycle_for_dataset "$data_name") || return 1
        enc_in=$(awk -F, 'NR == 1 { print NF - 1; exit }' "$data_file")

        for model in $MODELS
        do
            learning_rate=$(learning_rate_for_model "$model") || return 1
            for pred_len in $PRED_LENS
            do
                task_line="$model|$data_name|$data_stem|$enc_in|$cycle|$pred_len|$learning_rate"
                append_task_to_queue "$run_dir" "$task_index" "$task_line"
                task_index=$((task_index + 1))
            done
        done
    done

    echo "$task_index" > "$run_dir/total_tasks"
}

run_worker() {
    run_dir=$1
    gpu_id=$2
    queue_file=$run_dir/queue_gpu${gpu_id}.tsv
    status_file=$run_dir/status_gpu${gpu_id}.tsv
    current_file=$run_dir/current_gpu${gpu_id}.txt
    results_csv=$run_dir/results_gpu${gpu_id}.csv
    task_log_dir=$run_dir/tasks

    mkdir -p "$task_log_dir" "$RESULTS_ROOT" "$CHECKPOINTS_ROOT"
    if [ "${BASELINE_APPEND_STATUS:-0}" = "1" ]; then
        touch "$status_file"
    else
        : > "$status_file"
    fi
    wait_for_prior_run "$current_file"

    while IFS='|' read -r model data_name data_stem enc_in cycle pred_len learning_rate
    do
        [ -n "$model" ] || continue

        model_id=${data_stem}_${SEQ_LEN}_${pred_len}
        setting_channel_tag=cd
        if echo "$model" | grep -q '^SolarNet'; then
            setting_channel_tag=cd_scmfull
        fi
        setting=${model_id}_${model}_custom_ftM_sl${SEQ_LEN}_pl${pred_len}_${setting_channel_tag}${TRAIN_RATIO_TAG}_cycle${cycle}_seed${RANDOM_SEED}
        task_name=${model}_${data_stem}_sl${SEQ_LEN}_pl${pred_len}_train${TASK_RATIO_TAG}_cycle${cycle}
        metrics_file=$RESULTS_ROOT/$setting/metrics.json
        extra_args=$(model_extra_args "$model")
        solar_args=
        if [ "$model" = "SolarNet" ] && [ -n "$SOLAR_LATITUDE" ] && [ -n "$SOLAR_LONGITUDE" ]; then
            solar_args="--latitude $SOLAR_LATITUDE --longitude $SOLAR_LONGITUDE --solar_timezone ${SOLAR_TIMEZONE:-UTC}"
        fi

        echo "$task_name" > "$current_file"
        if [ -f "$metrics_file" ]; then
            printf 'SKIPPED\t%s\t%s\n' "$(date -u +%FT%TZ)" "$task_name" >> "$status_file"
            continue
        fi

        task_completed=0
        task_terminal_failure=0
        for task_batch_size in $(batch_candidates)
        do
            task_log=$task_log_dir/${task_name}_bs${task_batch_size}.log
            printf 'RUNNING\t%s\t%s\tbatch=%s\n' "$(date -u +%FT%TZ)" "$task_name" "$task_batch_size" >> "$status_file"
            # extra_args is a controlled list emitted by model_extra_args.
            # shellcheck disable=SC2086
            if CUDA_VISIBLE_DEVICES=$gpu_id "$PYTHON" -u "$PROJECT_ROOT/run.py" \
                --is_training 1 \
                --root_path "$DATASET_DIR/" \
                --data_path "$data_name" \
                --model_id "$model_id" \
                --model "$model" \
                --data custom \
                --features M \
                --seq_len "$SEQ_LEN" \
                --pred_len "$pred_len" \
                --train_ratio "$TRAIN_RATIO" \
                --enc_in "$enc_in" \
                --dec_in "$enc_in" \
                --c_out "$enc_in" \
                --cycle "$cycle" \
                --train_epochs "$TRAIN_EPOCHS" \
                --patience "$PATIENCE" \
                --itr 1 \
                --batch_size "$task_batch_size" \
                --learning_rate "$learning_rate" \
                --random_seed "$RANDOM_SEED" \
                --num_workers "$NUM_WORKERS" \
                --gpu 0 \
                --checkpoints "$CHECKPOINTS_ROOT/" \
                --results_dir "$RESULTS_ROOT/" \
                --results_csv "$results_csv" \
                $extra_args \
                $solar_args > "$task_log" 2>&1
            then
                printf 'COMPLETED\t%s\t%s\tbatch=%s\n' "$(date -u +%FT%TZ)" "$task_name" "$task_batch_size" >> "$status_file"
                task_completed=1
                break
            else
                exit_code=$?
                if grep -Eq 'CUDA out of memory|OutOfMemoryError' "$task_log"; then
                    printf 'OOM_RETRY\t%s\t%s\tbatch=%s exit=%s\n' "$(date -u +%FT%TZ)" "$task_name" "$task_batch_size" "$exit_code" >> "$status_file"
                else
                    printf 'FAILED\t%s\t%s\tbatch=%s exit=%s\n' "$(date -u +%FT%TZ)" "$task_name" "$task_batch_size" "$exit_code" >> "$status_file"
                    task_terminal_failure=1
                    break
                fi
            fi
        done
        if [ "$task_completed" -eq 0 ] && [ "$task_terminal_failure" -eq 0 ]; then
            printf 'FAILED\t%s\t%s\tall_batch_sizes_exhausted\n' "$(date -u +%FT%TZ)" "$task_name" >> "$status_file"
        fi
    done < "$queue_file"

    echo "IDLE" > "$current_file"
    date -u +%FT%TZ > "$run_dir/done_gpu${gpu_id}"
}

show_status() {
    if [ $# -ge 1 ]; then
        run_dir=$1
    elif [ -f "$RUNS_ROOT/latest_run" ]; then
        run_dir=$(sed -n '1p' "$RUNS_ROOT/latest_run")
    else
        echo "No baseline run has been started."
        return 1
    fi

    echo "Run directory: $run_dir"
    if [ -f "$run_dir/total_tasks" ]; then
        echo "Total tasks: $(sed -n '1p' "$run_dir/total_tasks")"
    fi
    for gpu_id in 0 1
    do
        status_file=$run_dir/status_gpu${gpu_id}.tsv
        completed=0
        failed=0
        skipped=0
        [ -f "$status_file" ] && completed=$(awk -F '\t' '$1 == "COMPLETED" { count++ } END { print count + 0 }' "$status_file")
        [ -f "$status_file" ] && failed=$(awk -F '\t' '$1 == "FAILED" { count++ } END { print count + 0 }' "$status_file")
        [ -f "$status_file" ] && skipped=$(awk -F '\t' '$1 == "SKIPPED" { count++ } END { print count + 0 }' "$status_file")
        current=NOT_STARTED
        [ -f "$run_dir/current_gpu${gpu_id}.txt" ] && current=$(sed -n '1p' "$run_dir/current_gpu${gpu_id}.txt")
        service_state=NOT_STARTED
        if [ -f "$run_dir/unit_gpu${gpu_id}.txt" ]; then
            unit_name=$(sed -n '1p' "$run_dir/unit_gpu${gpu_id}.txt")
            service_state=$(systemctl --user is-active "$unit_name" 2>/dev/null || true)
        fi
        echo "GPU $gpu_id: service=$service_state completed=$completed failed=$failed skipped=$skipped current=$current"
    done
}

start_run() {
    if [ ! -x "$PYTHON" ]; then
        echo "Python environment not found at $PYTHON. Run 'uv sync' first." >&2
        return 1
    fi

    timestamp=$(date -u +%Y%m%dT%H%M%SZ)
    run_dir=$RUNS_ROOT/$timestamp
    mkdir -p "$run_dir"
    build_queues "$run_dir" || return 1
    echo "$run_dir" > "$RUNS_ROOT/latest_run"

    for gpu_id in 0 1
    do
        if [ -n "$UNIT_TAG" ]; then
            unit_name=solarnet-baselines-${UNIT_TAG}-${timestamp}-gpu${gpu_id}
        else
            unit_name=solarnet-baselines-${timestamp}-gpu${gpu_id}
        fi
        echo "$unit_name" > "$run_dir/unit_gpu${gpu_id}.txt"
        systemd-run --user \
            --unit "$unit_name" \
            --description "SolarNet baseline worker on GPU $gpu_id" \
            --working-directory "$PROJECT_ROOT" \
            /usr/bin/env \
            "BASELINE_PYTHON=$PYTHON" \
            "BASELINE_DATASET_DIR=$DATASET_DIR" \
            "BASELINE_RESULTS_DIR=$RESULTS_ROOT" \
            "BASELINE_CHECKPOINTS_DIR=$CHECKPOINTS_ROOT" \
            "BASELINE_RUNS_DIR=$RUNS_ROOT" \
            "BASELINE_SEQ_LEN=$SEQ_LEN" \
            "BASELINE_PRED_LENS=$PRED_LENS" \
            "BASELINE_RANDOM_SEED=$RANDOM_SEED" \
            "BASELINE_TRAIN_EPOCHS=$TRAIN_EPOCHS" \
            "BASELINE_TRAIN_RATIO=$TRAIN_RATIO" \
            "BASELINE_PATIENCE=$PATIENCE" \
            "BASELINE_BATCH_SIZE=$BATCH_SIZE" \
            "BASELINE_BATCH_FALLBACKS=$BATCH_FALLBACKS" \
            "BASELINE_NUM_WORKERS=$NUM_WORKERS" \
            "BASELINE_DATASET_PATTERN=$DATASET_PATTERN" \
            "BASELINE_MODELS=$MODELS" \
            "BASELINE_SOLAR_LATITUDE=$SOLAR_LATITUDE" \
            "BASELINE_SOLAR_LONGITUDE=$SOLAR_LONGITUDE" \
            "BASELINE_SOLAR_TIMEZONE=$SOLAR_TIMEZONE" \
            "BASELINE_WAIT_FOR_RUN_DIR=$WAIT_FOR_RUN_DIR" \
            "BASELINE_RETRY_FAILED_FROM=$RETRY_FAILED_FROM" \
            /bin/sh "$0" worker "$run_dir" "$gpu_id"
    done

    echo "Started baseline workers."
    show_status "$run_dir"
}

command=${1:-}
case "$command" in
    start)
        start_run
        ;;
    worker)
        [ $# -eq 3 ] || { usage; exit 1; }
        run_worker "$2" "$3"
        ;;
    status)
        if [ $# -ge 2 ]; then
            show_status "$2"
        else
            show_status
        fi
        ;;
    *)
        usage
        exit 1
        ;;
esac
