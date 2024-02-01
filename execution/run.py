import os
import io
import types
import json
import argparse
import pickle
from functools import partial
from collections import defaultdict
from execution import evaluate_with_test_code, evaluate_with_test_inputs
from human_eval.data import read_problems
from utils import convert_test_case_to_input_output

parser = argparse.ArgumentParser()
parser.add_argument('--solution_path', type=str, help='')
parser.add_argument('--test_path', type=str, help='')
parser.add_argument('--benchmark_path', type=str, help='')
parser.add_argument('--out_path', type=str, help='')
parser.add_argument('--timeout', type=int, help='')
parser.add_argument('--limit', type=int, help='')

args = parser.parse_args()

os.makedirs(args.out_path, exist_ok=True)

with open(args.solution_path) as f:
    handled_solutions = [json.loads(line) for line in f]

with open(args.test_path) as f:
    handled_test_cases = [json.loads(line) for line in f]
    for item in handled_test_cases:
        item["test_cases"] = item["test_cases"][:args.limit]

raw_problems = read_problems(args.benchmark_path)

for i, item in enumerate(handled_solutions):
    task_id = item["task_id"]
    item["prompt"] = raw_problems[task_id]["prompt"]
    item["entry_point"] = raw_problems[task_id]["entry_point"]
    item["test"] = raw_problems[task_id]["test"]
    item['solution_id'] = i

model_generated_test_cases = defaultdict(dict)
test_inputs_dict = defaultdict(list)
count_exception = 0
count_total = 0
for item in handled_test_cases:
    task_id = item["task_id"]
    prompt = raw_problems[task_id]["prompt"]
    test_cases = item["test_cases"]
    entry_point = raw_problems[task_id]["entry_point"]
    input_output_list = list(map(partial(convert_test_case_to_input_output, prompt=prompt, entry_point=entry_point), test_cases))

    inputs, outputs = [], []
    for input, output in input_output_list:
        count_total += 1
        if input != "":
            inputs.append(input)
            outputs.append(output)
        else:
            count_exception += 1

    test_inputs_dict[task_id].append(inputs)

    exec(prompt+"\n    pass")

    for i, (input, output) in enumerate(zip(inputs, outputs)):
        test_case = test_cases[i]
        try:
            output = eval(output)
        except (SyntaxError, ValueError, NameError, TypeError, ZeroDivisionError, KeyError, OverflowError, IndexError, LookupError, AttributeError):
            count_exception += 1
            continue
        if isinstance(output, type({}.keys())):
            count_exception += 1
            continue
        if isinstance(output, type({}.values())):
            count_exception += 1
            continue
        if isinstance(output, type({}.items())):
            count_exception += 1
            continue
        if isinstance(output, types.GeneratorType):
            count_exception += 1
            continue
        if isinstance(output, io.IOBase):
            count_exception += 1
            continue
        if isinstance(output, types.FunctionType):
            print(f"WARNING: {output = } is a function")
            count_exception += 1
            continue
        
        current_outputs = model_generated_test_cases[task_id].get(input, [])
        current_outputs.append(output)
        model_generated_test_cases[task_id][input] = current_outputs

print(f"{count_total = }, {count_exception = }, percentage = {round(100*(count_exception/count_total), 2)}%")

ground_truth_exec_result = evaluate_with_test_code(handled_solutions, timeout=args.timeout)
with open(os.path.join(args.out_path, "ground_truth_exec_result.pkl"), "wb") as f:
    pickle.dump(ground_truth_exec_result, f) 

test_inputs_exec_result = evaluate_with_test_inputs(handled_solutions, test_inputs_dict, timeout=args.timeout, limit=args.limit)
with open(os.path.join(args.out_path, "test_inputs_exec_result.pkl"), "wb") as f:
    pickle.dump(test_inputs_exec_result, f)

with open(os.path.join(args.out_path, "model_generated_test_cases.pkl"), "wb") as f:
    pickle.dump(model_generated_test_cases, f)
