#!/bin/bash

seq=${SEQ:-16}
subset=${SUBSET:--1}
dataset_tokens_max=${DS_TOKENSMAX:-512}
dataset_nsamples=${DS_NSAMPLES:-500000}
dataset_templates_base=${DS_TEMPL_BASE:-ours_v2}
dataset_seed=${DS_SEED:-72353534}
dataset_model=${DS_MODEL:-gpt-4.1-mini}
dataset_nexamples=${DS_NEXAMPLES:-100000}
dataset_offset_suffix=${DS_OFFSET:-_offset-400000}
dataset_logprobs_suffix=${DS_USELOGPROBS:-_use-logprobs}
model=${MODEL:-qurater_Qwen3-Reranker-0.6B-seq-cls_bsz512_lr5e-5_epochs2_warmup0.1_conf0.5_labeltemp1.0_ds-ours_v2-500000-200000-512-72353534-gpt-4.1-mini-logprobs}
checkpoint=${CHECKPOINT:-checkpoint-352}

header="uv run python ./scripts/run_inference_pairwise.py \
--model ./checkpoints-preferences/${model}/${checkpoint} \
--batch_size $seq \
--subset $subset \
./data/results/tokens_max_${dataset_tokens_max}\
/fwe-fortified_sampled-${dataset_nsamples}_seed-${dataset_seed}\
/${dataset_templates_base}\
/combined_${dataset_model}_nexamples-${dataset_nexamples}\
${dataset_offset_suffix}\
${dataset_logprobs_suffix} \
./data/annotations/${model}/tokens_max_${dataset_tokens_max}\
/fwe-fortified_sampled-${dataset_nsamples}_seed-${dataset_seed}\
/${dataset_templates_base}\
/combined_${dataset_model}_nexamples-${dataset_nexamples}\
${dataset_offset_suffix}${dataset_logprobs_suffix}.json
"

echo command: "${header}"
${header} 2>&1 | tee -a ./logs/inference-log.out