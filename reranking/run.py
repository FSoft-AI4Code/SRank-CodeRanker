import argparse
from evaluator import (Evaluator,
                       AttentionRankingEvaluator)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['attention', 'random'], default='attention')
    parser.add_argument('--ground_truth_exec_path', type=str)
    parser.add_argument('--test_inputs_exec_path', type=str)
    parser.add_argument('--model_generated_test_path', type=str)
    parser.add_argument('--logprob_path', type=str)
    parser.add_argument('--reverse_logprob_path', type=str)
    parser.add_argument("--dataset_name", type=str, choices=["HumanEval", "MbppEval"])
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--no-rejection", action="store_true", default=False)
    args = parser.parse_args()
    if args.method == "attention":
        evaluator = AttentionRankingEvaluator(args.dataset_name,
                                   args.ground_truth_exec_path,
                                   args.test_inputs_exec_path,
                                   args.model_generated_test_path,
                                   logprob_path=args.logprob_path,
                                   reverse_logprob_path=args.reverse_logprob_path,
                                   )
        print("pass rates by SRank reranking", evaluator.get_pass_k_results())
    elif args.method == "random":
        evaluator = Evaluator(args.dataset_name,
                                   args.ground_truth_exec_path,
                                   args.test_inputs_exec_path,
                                   args.model_generated_test_path,
                                   logprob_path=args.logprob_path,
                                   reverse_logprob_path=args.reverse_logprob_path,
                                   )
    print("pass rates by random selection: {}".format(evaluator.get_pass_k_results_of_random_selection()))
