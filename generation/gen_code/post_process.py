import re
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
from human_eval.data import read_problems, write_jsonl, stream_jsonl

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

STOP = ['\nclass', '\ndef', '\nif', '# Example usage', '# Test your code', '# Test Cases', '# Test cases', '# test the code', "# Example usage", "# example usage", "if __name__ ==", "\n>>>", "\n```"]

def postprocess_code_solution(prediction, entry_point):
    codes = re.split("def {}\(.*?\).*?:".format(entry_point), prediction)[1:]
    if len(codes) == 0:
        return ""
    for i, code in enumerate(codes):
        lines = code.split("\n")
        filtered_lines = []
        for line in lines:
            if line.startswith("#"):
                continue
            if line.startswith("print("):
                continue
            filtered_lines.append(line)
        code = "\n".join(filtered_lines)
        code = re.sub("'''.*?'''", "", code, flags=re.DOTALL)
        code = re.sub('""".*?"""', "", code, flags=re.DOTALL)
        if re.search("(^|\n)\w", code):
            code = code[:re.search("(^|\n)\w", code).start()]
        for stop in STOP:
            if stop in code:
                next_line = code.index(stop)
                code = code[:next_line]
        code = "\n".join([line for line in code.split("\n") if line.strip() != ""])
        codes[i] = code
    code = codes[np.argmax([len(code) for code in codes])]
    return code

files = sorted(glob(args.path + "/*.jsonl"))
if "human_eval" in args.benchmark_path:
    assert len(files) == 164, len(files)
elif "mbpp" in args.benchmark_path:
    assert len(files) == 427, len(files)
print("{} files in {}".format(len(files), args.path))

problems = read_problems(args.benchmark_path)

output = []

for code_file in tqdm(files, total=len(files)):
    codes = [c for c in stream_jsonl(code_file)]
    for code in codes:
        task_id = code['task_id']
        entry_point = problems[task_id]['entry_point']
        function_header = 'def {}\(.*\).*:\n'.format(entry_point)

        if "completion" in code:
            completion = code["all_code"].split("### Response:", maxsplit=1)[1]
            # completion = code["completion"]
        else:
            completion = code["all_code"]

        completion = completion.replace("\t", "    ")
        completion = postprocess_code_solution(completion, entry_point)

        code['completion'] = completion
    
    output += codes 
    
print("save to {}".format(args.out_path))
write_jsonl(args.out_path, output)
