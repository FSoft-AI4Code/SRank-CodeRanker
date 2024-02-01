# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import statistics
import numpy as np
from collections import defaultdict
import logging
from typing import List, Union
import itertools

logging.basicConfig(
    format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

def _turn_solution_scores_into_choose_count(sorted_passed_results, topk):
    result = []
    last_score = sorted_passed_results[0][1]
    merged_passed_and_scores = [sorted_passed_results[0]]
    for passed, score in sorted_passed_results[1:]:
        if score == last_score:
            last_passed = merged_passed_and_scores[-1][0]
            merged_passed_and_scores[-1] = (last_passed + passed, score)
        else:
            merged_passed_and_scores.append((passed, score))
            last_score = score
    for passed_and_scores in merged_passed_and_scores:
        result.append((passed_and_scores[0], 1))  # choose one from solutions_and_score

    if len(result) >= topk:
        return result[:topk]
    else:
        intial_choose_count = [1]*len(result)
        for i in range(topk-len(result)):
            intial_choose_count[i%len(result)] += 1
        for i, choose_count in enumerate(intial_choose_count):
            result[i] = (result[i][0], choose_count)
        return result
    

def get_result_of_sorted_solutions(sorted_results_by_task, topks=[1]):
    topk_results = dict()
    for topk in topks:
        pass_rates = []
        for task_id, passed_results in sorted_results_by_task.items():
            all_wrong_probability = 1
            solutions_and_probability = _turn_solution_scores_into_choose_count(passed_results, topk)
            for passed_results, choose_count in solutions_and_probability:
                current_wrong_prob = _estimator(len(passed_results), sum(passed_results), 1)
                repeat_current_wrong_prob = pow(current_wrong_prob, choose_count)
                all_wrong_probability *= repeat_current_wrong_prob
            pass_rates.append(1-all_wrong_probability)
        
        # the avg rate of all tasks
        topk_results[f'pass@{topk}'] = round(statistics.mean(pass_rates), 4)
    # logger.info(topk_results)
    return topk_results

def pass_at_K_by_task(results, k):
    result_dict = defaultdict(list)
    for line in results:
        result_dict[line['task_id']].append(line['passed'])
    result = dict()
    for task_id in result_dict.keys():
        total = len(result_dict[task_id])
        correct = sum(result_dict[task_id])
        score = _estimate_pass_at_k(total, [correct], k)[0]
        result[task_id] = score
    return result

def pass_at_K(results, k = [1, 10, 100]):
    # Calculate pass@k.
    total, correct = [], []
    for passed in results:
        total.append(len(passed))
        correct.append(sum(passed))

    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": round(_estimate_pass_at_k(total, correct, k).mean(), 4)
                 for k in ks if (total >= k).all()}

    return pass_at_k

def _estimator(n: int, c: int, k: int) -> float:
    """
    Calculates comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 0
    return np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def _estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """
    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([1.0 - _estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])
