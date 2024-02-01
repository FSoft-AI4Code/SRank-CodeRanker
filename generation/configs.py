import argparse

def add_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help="")
    parser.add_argument('--benchmark_path', type=str, default='../../../reranking/dataset/human_eval/dataset/CodeTHumanEval.jsonl', help="")
    parser.add_argument('--output_path', type=str, help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--temperature', type=float, default=0.8, help="")
    parser.add_argument('--N', type=int, default=200, help="")
    parser.add_argument('--max_len', type=int, default=600, help="")
    parser.add_argument('--decoding_style', type=str, default='sampling', help="")
    parser.add_argument('--num_seqs_per_iter', type=int, default=50, help='')
    parser.add_argument('--greedy_decode', action='store_true', help='')
    parser.add_argument('--overwrite', action='store_true', help='')

    args = parser.parse_args()

    return args
