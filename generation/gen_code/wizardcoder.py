import pprint
import sys
sys.path.append("..")
import os
import re
import math
from tqdm import tqdm
import torch
from transformers import GenerationConfig
from human_eval.data import write_jsonl, read_problems, stream_jsonl
from model_utils import get_model
from configs import add_args
from prompt import generate_prompt

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def main():
    args = add_args()

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    STOP_SEQS = ['\nclass', '\ndef', '\n#', '\nif', '\nprint']

    problems = read_problems(args.benchmark_path)

    task_ids = sorted(problems.keys())[args.start_index: args.end_index]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids]
    num_samples = len(prompts)
    print("Number of samples: {}".format(num_samples))

    tokenizer, model = get_model(base_model=args.model, device=device)
    if args.greedy_decode:
        generation_config = GenerationConfig(
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            max_length=args.max_len,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        generation_config = GenerationConfig(
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=args.temperature,
            max_length=args.max_len,
            num_return_sequences=args.num_seqs_per_iter,
            eos_token_id=tokenizer.eos_token_id,
            top_p=0.95
        )

    print(f"Loaded {args.model}.")
    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i)

        if os.path.exists(output_file) and not args.overwrite:
            print(f'Skip {output_file} as it already exists')
            continue

        prompt = prompts[i].replace('    ', '\t')
        prompt_batch = [generate_prompt(prompt, args.model)]

        ids_batch = [task_ids[i]]

        completion_seqs = []

        encoding = tokenizer(prompt_batch, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)

        if args.decoding_style == 'sampling':
            loops = math.ceil(args.N / args.num_seqs_per_iter)
        else:
            loops = 2

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

            with torch.no_grad():
                gen_tokens = model.generate(
                    **encoding,
                    generation_config=generation_config
                )

            if gen_tokens is not None:
                gen_seqs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            else:
                gen_seqs = None

            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[1]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

                    completion_seqs.append(
                        {'task_id': task_id,
                         'completion': completion_seq,
                         'all_code': all_code,
                         }
                    )

        print("Saving results to {}".format(output_file))
        write_jsonl(output_file, completion_seqs)

if __name__ == '__main__':
    main()
