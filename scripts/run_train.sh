#!/bin/bash

# > Default arguments - can be overriden by environment variables:
# architecture to train, must be compatible with the Llama architecture
model=${MODEL:-princeton-nlp/Sheared-LLaMA-1.3b}
# total batch size across all devices with gradient accumulation
bsz=${BSZ:-512}
# number of sequences per device
seq=${SEQ:-16}
# peak learning rate
lr=${LR:-5e-5}
# number of epochs
epochs=${EPOCHS:-20}
# warmup ratio
warmup=${WARMUP:-0.1}
# save model every n steps
save_steps=${SAVE:-200}
# suffix to append to run name
suffix=${SUFFIX:-""}
# only predict labels with certain confidence
confidence=${CONFIDENCE:-0.5}
# temperature applied to labels
labeltemp=${LABELTEMP:-1.0}
# which labels to predict
label_index=${LABELINDEX:-"all"}

dataset_templates_base=${DS_TEMPL_BASE:-ours_v2}
dataset_nsamples=${DS_NSAMPLES:-500000}
dataset_nexamples=${DS_NEXAMPLES:-20000}
dataset_tokens_max=${DS_TOKENSMAX:-2048}
dataset_seed=${DS_SEED:-72353534}
dataset_model=${DS_MODEL:-gpt-4.1-mini}
dataset_logprobs=${DS_USELOGPROBS:-logprobs}

run_name="qurater_$(basename $model)_bsz${bsz}_lr${lr}_epochs${epochs}_warmup${warmup}_conf${confidence}_labeltemp${labeltemp}${suffix}_ds-${dataset_templates_base}-${dataset_nsamples}-${dataset_nexamples}-${dataset_tokens_max}-${dataset_seed}-${dataset_model}-${dataset_logprobs}"
out_dir="checkpoints-preferences/$run_name"
mkdir -p $out_dir

nvidia-smi

# Determine available number of GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    num_gpus=$(nvidia-smi -L | wc -l)
else
    num_gpus=$(jq -n "[$CUDA_VISIBLE_DEVICES] | length")
fi
num_gpus=${NUM_GPUS:-$num_gpus}

# Determine number of nodes if run inside slurm job job
num_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
if [ $num_nodes == 0 ]; then
    num_nodes=1
fi
num_nodes=${NUM_NODES:-$num_nodes}

if [ $num_nodes -gt 1 ]; then
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    master_addr=${MASTER_ADDR:-$master_addr}
    master_port=${MASTER_PORT:-56321}

    header="uv run srun torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$master_addr:56321 \
    --nnodes=$num_nodes \
    --nproc-per-node=$num_gpus \
    -m qurating.test_train"
else
    master_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
    master_port=${master_port:-56321}
    master_port=${MASTER_PORT:-$master_port}

    header="uv run torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:$master_port \
    --nnodes=1 \
    --nproc-per-node=$num_gpus \
    -m qurating.test_train"
fi

export OMP_NUM_THREADS=$num_gpus

export WANDB_DIR=$out_dir
export WANDB_MODE="online"

export FSDP_SHARDING_STRATEGY="5" # 5 corresponds to _hybrid_shard_zero2
export FSDP_STATE_DICT_TYPE="FULL_STATE_DICT"

# labels=(
#     writing_style_average
#     required_expertise_average
#     facts_and_trivia_average
#     educational_value_average
# )
# if [[ $label_index == "all" ]]; then
#     label_field="${labels[@]}"
# else
#     label_field="${labels[$label_index]}"
# fi

base_arguments=(
    --report_to wandb

    # --do_eval
    # --do_train
    --model-name $model
    --dataset-samples $dataset_nsamples
    --dataset-seed $dataset_seed
    --dataset-templates $dataset_templates_base
    --dataset-max-tokens $dataset_tokens_max
    --judgement-model $dataset_model
    --num-examples $dataset_nexamples
    --logprobs $dataset_logprobs

    --output-dir $out_dir
    # --log_level info
    # --logging_steps 1
    # --disable_tqdm true
    --save-steps $save_steps
    # --evaluation_strategy steps
    # --eval-steps $save_steps
    # --load_best_model_at_end true
    # --metric_for_best_mode eval_validation_acc
    # --greater_is_better true
    # --dataloader_num_workers 2
    # --cache_dir .cache
    # --overwrite_output_dir
    # --remove_unused_columns false
    # --use_fast_tokenizer false

    --num_gpus $num_gpus
    --num_nodes $num_nodes

    --epochs $epochs
    --max-length 2048
    --batch-size-total $bsz
    --batch-size-per-device $seq
    # --gradient_accumulation_steps $(($bsz / $seq / $num_gpus / $num_nodes))
    --learning-rate $lr
    # --max_grad_norm 1.0
    # --weight_decay 0.1
    # --warmup_ratio $warmup

    # --bf16_full_eval
    # --bf16
    # --ddp_find_unused_parameters false
    # --fsdp auto_wrap

    # Depending on model size and sequence length, gradient checkpointing might result in higher throughput
    # --gradient_checkpointing

    # --label_field $label_field
    --confidence-threshold $confidence
    # --label_temperature $labeltemp

    # --train_datasets princeton-nlp/QuRating-GPT3.5-Judgments
    # --eval_split_size_train 0.1

    # --eval_datasets princeton-nlp/QuRating-GPT3.5-Judgments-Test
    # --eval_split_size 1.0

    $@
)

echo command: "${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}" 2>&1 | tee -a $out_dir/log.out
