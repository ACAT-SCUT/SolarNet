#!/bin/sh

set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

PYTHON=${TIMEBASE_PYTHON:-$PROJECT_ROOT/.venv/bin/python}
DATASET_DIR=${TIMEBASE_DATASET_DIR:-$PROJECT_ROOT/dataset}
RESULTS_ROOT=${TIMEBASE_RESULTS_DIR:-$PROJECT_ROOT/results/timebase_period288}
CHECKPOINTS_ROOT=${TIMEBASE_CHECKPOINTS_DIR:-$PROJECT_ROOT/checkpoints/timebase_period288}
RUNS_ROOT=${TIMEBASE_RUNS_DIR:-$PROJECT_ROOT/logs/timebase_period288}
PERIOD_LEN=${TIMEBASE_PERIOD_LEN:-288}
PRED_LENS=${TIMEBASE_PRED_LENS:-"12 24 48 96 144 288"}
DATASET_PATTERN=${TIMEBASE_DATASET_PATTERN:-"[ano]*[!1]5min.csv"}
TRAIN_EPOCHS=${TIMEBASE_TRAIN_EPOCHS:-30}
PATIENCE=${TIMEBASE_PATIENCE:-5}
BATCH_SIZE=${TIMEBASE_BATCH_SIZE:-64}
BATCH_FALLBACKS=${TIMEBASE_BATCH_FALLBACKS:-"32 16"}
NUM_WORKERS=${TIMEBASE_NUM_WORKERS:-8}
WAIT_GPU1=${TIMEBASE_WAIT_GPU1:-}

usage() {
    echo "Usage: sh scripts/Main/TimeBase_all.sh {start|worker RUN_DIR GPU_ID|status [RUN_DIR]}"
}

wait_for_run() {
    wait_dir=$1
    current_file=$2
    [ -n "$wait_dir" ] || return 0
    echo "WAITING_FOR $wait_dir" > "$current_file"
    while :
    do
        active=0
        for unit_file in "$wait_dir"/unit_gpu*.txt
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

build_queues() {
    run_dir=$1
    : > "$run_dir/queue_gpu0.tsv"
    : > "$run_dir/queue_gpu1.tsv"
    task_index=0
    for data_file in "$DATASET_DIR"/$DATASET_PATTERN
    do
        [ -f "$data_file" ] || continue
        data_name=$(basename "$data_file")
        data_stem=${data_name%.csv}
        enc_in=$(awk -F, 'NR == 1 { print NF - 1; exit }' "$data_file")
        for pred_len in $PRED_LENS
        do
            task_line="$data_name|$data_stem|$enc_in|$pred_len"
            # GPU 0 starts immediately, so give it two thirds of the queue.
            if [ $((task_index % 3)) -eq 2 ]; then
                echo "$task_line" >> "$run_dir/queue_gpu1.tsv"
            else
                echo "$task_line" >> "$run_dir/queue_gpu0.tsv"
            fi
            task_index=$((task_index + 1))
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
    : > "$status_file"
    if [ ! -s "$queue_file" ]; then
        echo "IDLE" > "$current_file"
        date -u +%FT%TZ > "$run_dir/done_gpu${gpu_id}"
        return 0
    fi
    if [ "$gpu_id" -eq 1 ]; then
        wait_for_run "$WAIT_GPU1" "$current_file"
    fi

    while IFS='|' read -r data_name data_stem enc_in pred_len
    do
        [ -n "$data_name" ] || continue
        task_name=TimeBase_${data_stem}_sl144_pl${pred_len}_period${PERIOD_LEN}
        model_id=${data_stem}_144_${pred_len}
        setting=${model_id}_TimeBase_custom_ftM_sl144_pl${pred_len}_cd_period${PERIOD_LEN}_cycle288_seed2024
        metrics_file=$RESULTS_ROOT/$setting/metrics.json
        echo "$task_name" > "$current_file"

        if [ -f "$metrics_file" ]; then
            printf 'SKIPPED\t%s\t%s\n' "$(date -u +%FT%TZ)" "$task_name" >> "$status_file"
            continue
        fi

        completed=0
        terminal_failure=0
        for task_batch_size in $BATCH_SIZE $BATCH_FALLBACKS
        do
            task_log=$task_log_dir/${task_name}_bs${task_batch_size}.log
            printf 'RUNNING\t%s\t%s\tbatch=%s\n' "$(date -u +%FT%TZ)" "$task_name" "$task_batch_size" >> "$status_file"
            if CUDA_VISIBLE_DEVICES=$gpu_id "$PYTHON" -u "$PROJECT_ROOT/run.py" \
                --is_training 1 \
                --root_path "$DATASET_DIR/" \
                --data_path "$data_name" \
                --model_id "$model_id" \
                --model TimeBase \
                --data custom \
                --features M \
                --seq_len 144 \
                --pred_len "$pred_len" \
                --enc_in "$enc_in" \
                --dec_in "$enc_in" \
                --c_out "$enc_in" \
                --cycle 288 \
                --period_len "$PERIOD_LEN" \
                --basis_num 6 \
                --use_period_norm 1 \
                --use_orthogonal 0 \
                --orthogonal_weight 0.2 \
                --individual 0 \
                --train_epochs "$TRAIN_EPOCHS" \
                --patience "$PATIENCE" \
                --itr 1 \
                --batch_size "$task_batch_size" \
                --learning_rate 0.001 \
                --random_seed 2024 \
                --num_workers "$NUM_WORKERS" \
                --gpu 0 \
                --checkpoints "$CHECKPOINTS_ROOT/" \
                --results_dir "$RESULTS_ROOT/" \
                --results_csv "$results_csv" > "$task_log" 2>&1
            then
                printf 'COMPLETED\t%s\t%s\tbatch=%s\n' "$(date -u +%FT%TZ)" "$task_name" "$task_batch_size" >> "$status_file"
                completed=1
                break
            else
                exit_code=$?
                if grep -Eq 'CUDA out of memory|OutOfMemoryError' "$task_log"; then
                    printf 'OOM_RETRY\t%s\t%s\tbatch=%s exit=%s\n' "$(date -u +%FT%TZ)" "$task_name" "$task_batch_size" "$exit_code" >> "$status_file"
                else
                    printf 'FAILED\t%s\t%s\tbatch=%s exit=%s\n' "$(date -u +%FT%TZ)" "$task_name" "$task_batch_size" "$exit_code" >> "$status_file"
                    terminal_failure=1
                    break
                fi
            fi
        done
        if [ "$completed" -eq 0 ] && [ "$terminal_failure" -eq 0 ]; then
            printf 'FAILED\t%s\t%s\tall_batch_sizes_exhausted\n' "$(date -u +%FT%TZ)" "$task_name" >> "$status_file"
        fi
    done < "$queue_file"

    echo "IDLE" > "$current_file"
    date -u +%FT%TZ > "$run_dir/done_gpu${gpu_id}"
}

show_status() {
    if [ $# -ge 1 ]; then
        run_dir=$1
    else
        run_dir=$(sed -n '1p' "$RUNS_ROOT/latest_run")
    fi
    echo "Run directory: $run_dir"
    echo "Total tasks: $(sed -n '1p' "$run_dir/total_tasks")"
    for gpu_id in 0 1
    do
        status_file=$run_dir/status_gpu${gpu_id}.tsv
        completed=0
        failed=0
        [ -f "$status_file" ] && completed=$(awk -F '\t' '$1 == "COMPLETED" { n++ } END { print n + 0 }' "$status_file")
        [ -f "$status_file" ] && failed=$(awk -F '\t' '$1 == "FAILED" { n++ } END { print n + 0 }' "$status_file")
        current=NOT_STARTED
        [ -f "$run_dir/current_gpu${gpu_id}.txt" ] && current=$(sed -n '1p' "$run_dir/current_gpu${gpu_id}.txt")
        service=NOT_STARTED
        if [ -f "$run_dir/unit_gpu${gpu_id}.txt" ]; then
            unit=$(sed -n '1p' "$run_dir/unit_gpu${gpu_id}.txt")
            service=$(systemctl --user is-active "$unit" 2>/dev/null || true)
        fi
        echo "GPU $gpu_id: service=$service completed=$completed failed=$failed current=$current"
    done
}

start_run() {
    [ -x "$PYTHON" ] || { echo "Python not found: $PYTHON" >&2; return 1; }
    timestamp=$(date -u +%Y%m%dT%H%M%SZ)
    run_dir=$RUNS_ROOT/$timestamp
    mkdir -p "$run_dir"
    build_queues "$run_dir"
    echo "$run_dir" > "$RUNS_ROOT/latest_run"
    for gpu_id in 0 1
    do
        unit=timebase-${timestamp}-gpu${gpu_id}
        echo "$unit" > "$run_dir/unit_gpu${gpu_id}.txt"
        systemd-run --user \
            --unit "$unit" \
            --description "TimeBase worker on GPU $gpu_id" \
            --working-directory "$PROJECT_ROOT" \
            /usr/bin/env \
            "TIMEBASE_PYTHON=$PYTHON" \
            "TIMEBASE_DATASET_DIR=$DATASET_DIR" \
            "TIMEBASE_RESULTS_DIR=$RESULTS_ROOT" \
            "TIMEBASE_CHECKPOINTS_DIR=$CHECKPOINTS_ROOT" \
            "TIMEBASE_RUNS_DIR=$RUNS_ROOT" \
            "TIMEBASE_PERIOD_LEN=$PERIOD_LEN" \
            "TIMEBASE_TRAIN_EPOCHS=$TRAIN_EPOCHS" \
            "TIMEBASE_PATIENCE=$PATIENCE" \
            "TIMEBASE_BATCH_SIZE=$BATCH_SIZE" \
            "TIMEBASE_BATCH_FALLBACKS=$BATCH_FALLBACKS" \
            "TIMEBASE_NUM_WORKERS=$NUM_WORKERS" \
            "TIMEBASE_WAIT_GPU1=$WAIT_GPU1" \
            /bin/sh "$0" worker "$run_dir" "$gpu_id"
    done
    show_status "$run_dir"
}

command=${1:-}
case "$command" in
    start) start_run ;;
    worker) [ "$#" -eq 3 ] || { usage; exit 1; }; run_worker "$2" "$3" ;;
    status) if [ "$#" -ge 2 ]; then show_status "$2"; else show_status; fi ;;
    *) usage; exit 1 ;;
esac
