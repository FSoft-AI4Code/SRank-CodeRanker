devices=$1
model_name="wizardcoder34B"
model="wizardLM/WizardCoder-Python-34B-V1.0"
dataset=$2
temp=$3
max_len=2048
pred_num=100
num_seqs_per_iter=2

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

cd ../
output_path=preds/${dataset}/${model_name}/T${temp}_N${pred_num}
mkdir -p ${output_path}
echo 'Output path: '$output_path
echo 'Model to eval: '$model

IFS=',' read -ra gpus <<< "$devices"
gpu_num=${#gpus[@]}

CUDA_VISIBLE_DEVICES=${devices} python wizardcoder34B.py --model ${model} --benchmark_path ${dataset_path} \
  --start_index 0 --end_index ${num_tasks} --temperature ${temp} \
  --num_seqs_per_iter ${num_seqs_per_iter} --N ${pred_num} \
  --max_len ${max_len} --output_path ${output_path} --num_gpus ${gpu_num}
