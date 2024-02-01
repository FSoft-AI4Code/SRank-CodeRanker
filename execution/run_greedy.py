import os
import json
import argparse
import pickle
import numpy as np
from execution import evaluate_with_test_code
from human_eval.data import read_problems

parser = argparse.ArgumentParser()
parser.add_argument('--solution_path', type=str, help='')
parser.add_argument('--benchmark_path', type=str, help='')
parser.add_argument('--out_path', type=str, help='')
parser.add_argument('--timeout', type=int, help='')
parser.add_argument('--limit', type=int, help='')

args = parser.parse_args()

os.makedirs(args.out_path, exist_ok=True)

with open(args.solution_path) as f:
    handled_solutions = [json.loads(line) for line in f]

raw_problems = read_problems(args.benchmark_path)

for i, item in enumerate(handled_solutions):
    task_id = item["task_id"]
    item["prompt"] = raw_problems[task_id]["prompt"]
    item["entry_point"] = raw_problems[task_id]["entry_point"]
    item["test"] = raw_problems[task_id]["test"]
    item['solution_id'] = i

ground_truth_exec_result = evaluate_with_test_code(handled_solutions, timeout=args.timeout)
with open(os.path.join(args.out_path, "ground_truth_exec_result.pkl"), "wb") as f:
    pickle.dump(ground_truth_exec_result, f) 

print("pass@1: {}".format(np.mean([item["passed"] for item in ground_truth_exec_result])))
