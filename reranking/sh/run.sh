#! /bin/bash

model=$1
dataset=$2
temp=$3
pred_num=$4
method=$5

if [ ${dataset} == "humaneval" ]; then
    dataset_name="HumanEval"
    dataset_path="../../reranking/dataset/human_eval/dataset/CodeTHumanEval.jsonl"
    num_tasks=164
    # pred_num=100
elif [ ${dataset} == "mbpp" ]; then
    dataset_name="MbppEval"
    dataset_path="../../reranking/dataset/mbpp/mbpp_sanitized_for_code_generation.jsonl"
    num_tasks=427
    # pred_num=100
elif [ ${dataset} == "apps" ]; then
    dataset_path="../../reranking/dataset/apps/APPS_zeroshot_for_code_generation.jsonl"
    num_tasks=5000
    # pred_num=50
else
    echo "Unknow dataset: $dataset"
    exit 1
fi

if [ ${model} == "coder_reviewer_codex002" ]; then
    logprob_path=../generation/extract_prob/preds/${dataset}/${model}/T${temp}_N${pred_num}/logprob.jsonl
    reverse_logprob_path=../generation/extract_prob/preds/${dataset}/${model}/T${temp}_N${pred_num}/reverse_logprob.jsonl
else
    logprob_path="None"
    reverse_logprob_path="None"
    # echo "Unknow dataset: $dataset"
    # exit 1
fi

ground_truth_exec_path=../execution/results/${dataset}/${model}/T${temp}_N${pred_num}/ground_truth_exec_result.pkl
test_inputs_exec_path=../execution/results/${dataset}/${model}/T${temp}_N${pred_num}/test_inputs_exec_result.pkl
model_generated_test_path=../execution/results/${dataset}/${model}/T${temp}_N${pred_num}/model_generated_test_cases.pkl
out_dir=results/${dataset}/${model}/T${temp}_N${pred_num}

cd ..

python -i run.py --method ${method} --ground_truth_exec_path ${ground_truth_exec_path} \
        --test_inputs_exec_path ${test_inputs_exec_path} \
        --model_generated_test_path ${model_generated_test_path} \
        --logprob_path ${logprob_path} \
        --reverse_logprob_path ${reverse_logprob_path} \
        --dataset_name ${dataset_name} \
        --verbose
