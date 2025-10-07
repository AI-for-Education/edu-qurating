#!/bin/bash

seq=${SEQ:-16}
subset=${SUBSET:--1}
dataset_tokens_max=${DS_TOKENSMAX:-512}
dataset_nsamples=${DS_NSAMPLES:-20000}
dataset_templates_base=${DS_TEMPL_BASE:-ours_v2}
dataset_seed=${DS_SEED:-72353534}
dataset_model=${DS_MODEL:-gpt-5-mini-2025-08-07-minimal}
dataset_nexamples=${DS_NEXAMPLES:-20000}
model=${MODEL:-qurater_Sheared-LLaMA-1.3b_bsz512_lr5e-5_epochs20_warmup0.1_conf0.5_labeltemp1.0}
checkpoint=${CHECKPOINT:-checkpoint-360}

header="uv run python ./scripts/run_inference_pairwise.py \
--model ./test_training_output/${model}/${checkpoint} \
--batch_size $seq \
--subset $subset \
./data/results/tokens_max_${dataset_tokens_max}\
/fwe-fortified_sampled-${dataset_nsamples}_seed-${dataset_seed}\
/${dataset_templates_base}\
/combined_${dataset_model}_nexamples-${dataset_nexamples}.parquet \
./data/annotations/${model}/tokens_max_${dataset_tokens_max}\
/fwe-fortified_sampled-${dataset_nsamples}_seed-${dataset_seed}\
/${dataset_templates_base}\
/combined_${dataset_model}_nexamples-${dataset_nexamples}.json
"

echo command: "${header}"
${header} 2>&1 | tee -a ./logs/inference-log.out