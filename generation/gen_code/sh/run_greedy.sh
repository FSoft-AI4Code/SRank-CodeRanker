devices=$1
model_name=$2
dataset=$3
max_len=$4
python_script=$5

if [ ${model_name} == "starcoder" ]; then
    model="bigcode/starcoder"
elif [ ${model_name} == "codegen25" ]; then
    model="Salesforce/codegen25-7b-instruct"
elif [ ${model_name} == "codet5p" ]; then
    model="Salesforce/instructcodet5p-16b"
elif [ ${model_name} == "wizardcoder" ]; then
    model="WizardLM/WizardCoder-15B-V1.0"
elif [ ${model_name} == "llama2" ]; then
    model="meta-llama/Llama-2-70b"
else
    echo "Unknow model: $model_name"
    exit 1
fi

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
output_path=preds/${dataset}/${model_name}/greedy_decode
mkdir -p ${output_path}

echo 'Dataset: '$dataset
echo 'Number of tasks: '$num_tasks
echo 'Output path: '$output_path
echo 'Model to eval: '$model

index=0
IFS=',' read -ra gpus <<< "$devices"
gpu_num=${#gpus[@]}
num_samples_per_gpu=$(($num_tasks / $gpu_num))
num_samples_per_gpu=$((num_samples_per_gpu + (num_tasks % gpu_num != 0)))

echo 'Number of samples per gpu: '$num_samples_per_gpu

for ((i = 0; i < $gpu_num; i++)); do
  start_index=$((i * $num_samples_per_gpu))
  end_index=$(((i + 1) * $num_samples_per_gpu))

  # gpu=$((i))
  gpu=${gpus[i]}
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
    CUDA_VISIBLE_DEVICES=$gpu python ${python_script} --model ${model} --benchmark_path ${dataset_path} \
      --greedy_decode --start_index ${start_index} --end_index ${end_index} --temperature 0 \
      --num_seqs_per_iter 1 --N 1 --max_len ${max_len} --output_path ${output_path}
  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done
