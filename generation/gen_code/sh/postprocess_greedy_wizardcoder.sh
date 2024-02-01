#! /bin/bash

model_name=$1
dataset=$2

if [ ${dataset} == "humaneval" ]; then
    dataset_path="../../reranking/dataset/human_eval/dataset/CodeTHumanEval.jsonl"
    num_tasks=164
elif [ ${dataset} == "mbpp" ]; then
    dataset_path="../../reranking/dataset/mbpp/mbpp_sanitized_for_code_generation.jsonl"
    num_tasks=427
elif [ ${dataset} == "apps" ]; then
    dataset_path="../../reranking/dataset/apps/APPS_zeroshot_for_code_generation.jsonl"
    num_tasks=5000
else
    echo "Unknow dataset: $dataset"
    exit 1
fi

path=preds/${dataset}/${model_name}/greedy_decode
out_path=preds/${dataset}/${model_name}/postprocessed_greedy.jsonl

cd ..

python -i postprocess_wizardcoder.py --path ${path} --out_path ${out_path} --add_prompt
