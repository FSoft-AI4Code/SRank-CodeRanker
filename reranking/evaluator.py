import json
import pickle
import collections
from functools import partial
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool
import numpy as np

from functionality_processing import process_cluster_attention_one_task
from codet_evaluation import get_result_of_sorted_solutions, pass_at_K

class Evaluator:
    def __init__(
        self,
        dataset_name,
        ground_truth_exec_path,
        test_inputs_exec_path,
        model_generated_test_path,
        logprob_path=None,
        reverse_logprob_path=None,
        verbose=False,
        no_rejection=False,
        num_limit_test_cases=int(1e6),
    ):
        assert dataset_name in ["HumanEval", "MbppEval"]
        self.dataset_name = dataset_name
        self.verbose = verbose

        print("Loading ground_truth_exec_path ...")
        with open(ground_truth_exec_path, "rb") as f:
            self.data = pickle.load(f)
        print("Finishing loading ground_truth_exec_path")

        print("Loading test_inputs_exec_path ...")
        with open(test_inputs_exec_path, "rb") as f:
            self.test_inputs_exec_results = pickle.load(f)
        print("Finishing loading test_input_exec_path")

        print("Loading model_generated_test_path ...")
        with open(model_generated_test_path, "rb") as f:
            self.model_generated_input_output_dicts = pickle.load(f)
        print("Finishing loading model_generated_test_path")

        if logprob_path != "None":
            print("Loading logprob path ...")
            with open(logprob_path) as f:
                self.logprob = [json.loads(line) for line in f]
            print("Finishing loading logprob_path") 

            print("Loading reverse logprob path ...")
            with open(reverse_logprob_path) as f:
                self.reverse_logprob = [json.loads(line) for line in f]
            print("Finishing loading reverse logprob path")
        else:
            self.logprob = None
            self.reverse_logprob = None

        # Parsing results
        self.parse_data()

        self.data_by_task_id = collections.defaultdict(list)
        for entry in self.data:
            self.data_by_task_id[entry["task_id"]].append(entry)

        self.num_samples_per_task = len(self.data_by_task_id[self.data[0]["task_id"]])
        if not all(len(task_entries) == self.num_samples_per_task for task_entries in self.data_by_task_id.values()):
            print("WARNING: all tasks do not have the same number of generated samples")
            min_num_samples = min(len(task_entries) for task_entries in self.data_by_task_id.values())
            self.num_samples_per_task = min_num_samples
            for task_id, task_entries in self.data_by_task_id.items():
                self.data_by_task_id[task_id] = task_entries[:min_num_samples]

        print(f"{self.num_samples_per_task} cached samples")

        self.num_limit_test_cases = num_limit_test_cases

    def parse_data(self):
        data_by_completion = collections.defaultdict(dict)
        for item in self.data:
            data_by_completion[item["task_id"]][item["completion"]] = item

        for item in self.test_inputs_exec_results:
            _item = data_by_completion[item["task_id"]][item["completion"]]
            _item["test_inputs"] = item["test_inputs"]
            _item["outputs"] = item["outputs"]
            _item["test_inputs_passed"] = item["passed"]

        logprob_fields = []
        if self.logprob is not None:
            logprob_fields = ["logprob", "reverse_logprob"]
            for item in self.logprob:
                _item = data_by_completion[item["task_id"]][item["completion"]]
                _item["logprob"] = item["logprob"]
                _item["avg_logprob"] = np.mean(item["logprob"])
                _item["sum_logprob"] = sum(item["logprob"])

            for item in self.reverse_logprob:
                _item = data_by_completion[item["task_id"]][item["completion"]]
                _item["reverse_logprob"] = item["reverse_logprob"]
                _item["avg_reverse_logprob"] = np.mean(item["reverse_logprob"])
                _item["sum_reverse_logprob"] = sum(item["reverse_logprob"])

        for i, item in enumerate(self.data):
            _item = data_by_completion[item["task_id"]][item["completion"]]
            self.data[i] = _item

        for i, item in enumerate(self.data):
            if "test_inputs" not in item:
                item["test_inputs"] = []
                item["outputs"] = []
                item["test_inputs_passed"] = []

        # double checking
        for i, item in enumerate(self.data):
            for key in ["test_inputs", "outputs", "test_inputs_passed", "passed"] + logprob_fields:
                assert key in item, f"{i}, {key}, {item['task_id']}\n\n{item['completion']}"

    def get_pass_k_results_of_random_selection(self):
        passed_results = []
        for task_items in self.data_by_task_id.values():
            task_passed_results = []
            for item in task_items:
                task_passed_results.append(item["passed"])
            passed_results.append(task_passed_results)

        pass_at_k = pass_at_K(passed_results, k=[1])
        return pass_at_k


class AttentionRankingEvaluator(Evaluator):
    def __init__(self,
            *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.with_attention = True
        self.weight_combination = (1,1)
        self.results_by_task_id = dict()
        self.process()

    def process(self):
        pbar = tqdm(total=len(self.data_by_task_id), desc="Process solution clustering")
        with Pool() as p:
            for task_id, attention_score_matrix, correctness_features, passed_results in \
                                                p.imap(partial(process_cluster_attention_one_task, num_limit_test_cases=self.num_limit_test_cases),
                                                [(task_id, task_items, self.model_generated_input_output_dicts[task_id]) \
                                                        for task_id, task_items in self.data_by_task_id.items()]
                                                ):
                self.results_by_task_id[task_id] = (attention_score_matrix, correctness_features, passed_results)
                pbar.update(1)

    def get_sorted_solutions(self):
        norm_num_solutions_weight, pass_rate_weight = self.weight_combination
        ranked_results = defaultdict(list)
        for task_id, (attention_score_matrix, correctness_features, passed_results) in self.results_by_task_id.items():
            V = np.array([(norm_num_solutions**norm_num_solutions_weight) * (pass_rate**pass_rate_weight) for norm_num_solutions, pass_rate in zip(*correctness_features)])
            if self.with_attention:
                ranking_scores = attention_score_matrix @ V
            else:
                ranking_scores = V
            ranking_scores = ranking_scores.tolist()
            if isinstance(ranking_scores, int):
                ranking_scores = [ranking_scores]
            passed_and_scores = [(passed_result, score) for passed_result, score in zip(passed_results, ranking_scores)]
            ranked_results[task_id] = sorted(passed_and_scores, key=lambda x: x[1], reverse=True)

        return ranked_results

    def get_pass_k_results(self):
        self.ranked_results = self.get_sorted_solutions()
        return get_result_of_sorted_solutions(self.ranked_results)
