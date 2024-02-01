import re
import json
import argparse
from collections import defaultdict
from glob import glob
from tqdm import tqdm
import numpy as np
from human_eval.data import read_problems, write_jsonl, stream_jsonl

STOP_TOKEN = ['\nclass', '\ndef', '\n#', '\nif', '\nprint']

def check_test_case_validation(test_case):
    if len(test_case.strip()) == 0:
        return False
    if "assert" not in test_case:
        return False
    try:
        multi_line_test_case = test_case.replace("\n", "\n    ")
        assert_in_a_block = f"try:\n    {multi_line_test_case}\nexcept:\n    pass"
        compile(assert_in_a_block, "", "exec")
        return True
    except:
        return False

def test_case_extract(content, entry_point):
    def _truncate(content):
        for identifier in STOP_TOKEN:
            if identifier in content:
                content = content.split(identifier)[0]
        return content.strip()
    
    split_by_assert = [f'assert {part}'.strip() for part in content.split('assert ')[1:] if (entry_point.strip() in part) and len(part.strip()) > 0]
    truncated_test_cases = [_truncate(i) for i in split_by_assert]
    checked_assertions = [i for i in truncated_test_cases if check_test_case_validation(i)]
    
    return checked_assertions

parser = argparse.ArgumentParser()

parser.add_argument(
        "--benchmark_path",
        type=str,
        help=""
        )
parser.add_argument(
        "--path",
        type=str,
        help=""
        )
parser.add_argument(
        "--out_path",
        type=str,
        help=""
        )

args = parser.parse_args()

files = sorted(glob(args.path + "/*.jsonl"))
print("{} files in {}".format(len(files), args.path))

problems = read_problems(args.benchmark_path)

output = []
model_name = "codegen" if "codegen" in args.path else "starcoder"
assert model_name in ["codegen", "starcoder"]

for code_file in tqdm(files, total=len(files)):
    codes = [c for c in stream_jsonl(code_file)]
    for code in codes:
        task_id = code['task_id']
        entry_point = problems[task_id]['entry_point']

        if "completion" in code:
            completion = code["completion"]
        else:
            completion = code["all_code"]

        test_cases = test_case_extract(completion, entry_point)
        code["test_cases"] = test_cases
    
    output += codes 
    
print("save to {}".format(args.out_path))
write_jsonl(args.out_path, output)
