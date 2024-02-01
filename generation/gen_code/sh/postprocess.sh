#! /bin/bash

model_name=$1
dataset=$2
temp=0.8


if [ ${dataset} == "humaneval" ]; then
    dataset_path="../../reranking/dataset/human_eval/dataset/CodeTHumanEval.jsonl"
    num_tasks=164
    pred_num=100
elif [ ${dataset} == "mbpp" ]; then
    dataset_path="../../reranking/dataset/mbpp/mbpp_sanitized_for_code_generation.jsonl"
    num_tasks=427
    pred_num=100
elif [ ${dataset} == "apps" ]; then
    dataset_path="../../reranking/dataset/apps/APPS_zeroshot_for_code_generation.jsonl"
    num_tasks=5000
    pred_num=50
else
    echo "Unknow dataset: $dataset"
    exit 1
fi

path=preds/${dataset}/${model_name}/T${temp}_N${pred_num}
out_path=preds/${dataset}/${model_name}/postprocessed_T${temp}_N${pred_num}.jsonl

cd ..

if [[ $model_name == *"wizardcoder"* ]]; then
    if [[ $dataset == "humaneval" ]]; then
        python -i postprocess_wizardcoder_humaneval.py --path ${path} --out_path ${out_path} --add_prompt
    elif [[ $dataset == "mbpp" ]]; then
        python -i postprocess_wizardcoder_mbpp.py --path ${path} --out_path ${out_path} --add_prompt --mbpp_path ${dataset_path}
    fi
else
    python -i post_process.py --benchmark_path ${dataset_path} --path ${path} --out_path ${out_path}
fi
