import numpy as np
from utils import check_validity

NUM_LIMIT_TEST_CASES = int(1e6)

def calculate_pair_overlap(outputs, other_outputs):
    assert len(outputs) == len(other_outputs), f"{len(outputs) = }, {len(other_outputs) = }, \n\n{outputs = }\n\n\n{other_outputs = }"

    if len(outputs) == 0:
        return 0

    overlap = 0
    num_inputs = len(outputs)

    for output, other_output in zip(outputs, other_outputs):
        try:
            overlap += output[0] and other_output[0] and output[1] == other_output[1]
        except TypeError:
            print(f"Cannot compare output and other_output\n\n{output[1] = }\n\n{other_output[1] = }",)
            continue
    
    overlap /= num_inputs

    return overlap

def calculate_cluster_pass_rate(outputs, model_generated_outputs):
    """
    Calculating distance between a cluster and virtual cluster.
    The returned number is normalized to be in range [0,1].
    Args:
        outputs List[List[Bool, Object]]
        other_outputs List[List[Object]]
    """
    assert len(outputs) == len(model_generated_outputs)

    if len(outputs) == 0:
        return 0

    num_inputs = 0
    pass_rate = 0

    for output, generated_outputs in zip(outputs, model_generated_outputs):
        for generated_output in generated_outputs:
            num_inputs += 1
            try:
                pass_rate += output[0] and output[1] == generated_output
            except TypeError:
                print(f"Cannot compare output to virtual_output\n\n{output[1] = }\n\n{generated_output = }")

    pass_rate /= num_inputs

    return pass_rate

def calculate_attention_score_matrix(cluster_execution_outputs):
    attention_score_matrix = []
    for i, outputs in enumerate(cluster_execution_outputs):
        pair_overlaps = []
        for j, other_outputs in enumerate(cluster_execution_outputs):
            if i == j:
                pair_overlaps.append(1)
            else:
                pair_overlap = calculate_pair_overlap(outputs, other_outputs)
                pair_overlaps.append(pair_overlap)

        attention_score_matrix.append(pair_overlaps)

    return np.array(attention_score_matrix) 

def process_cluster_attention_one_task(inputs, num_limit_test_cases=int(1e6)):
    task_id, task_items, model_generated_input_output_dict = inputs

    exist_completions = set()
    output_list_counter = []

    count_invalid = 0
    passed_results_of_invalid = []

    for i, item in enumerate(task_items):
        item["validity"] = True
        if not check_validity(item):
            item["validity"] = False
            count_invalid += 1
            passed_results_of_invalid.append(item["passed"])
            continue

        completion = item["completion"]
        outputs = item["outputs"][:num_limit_test_cases]
        passed = item["passed"]
        output_list_existed = False

        for i, counter in enumerate(output_list_counter):
            try:
                # if outputs == counter[0]:
                if all(output[0] == cluster_output[0] and output[1] == cluster_output[1] for output, cluster_output in zip(outputs, counter[0])):
                    output_list_existed = True
                    if not completion in exist_completions:
                        counter[1] += 1
                    counter[2].append(passed)
                    break
            except TypeError:
                print(f"Error comparing outputs to counter[0]\n\n{outputs = }\n\n{counter[0] = }")
                # TO-DO: adding passed results
                continue

        if not output_list_existed:
            output_list_counter.append([outputs, 1, [passed]])

        exist_completions.add(completion)

    # Calculate attention_score_matrix
    cluster_execution_outputs = [counter[0] for counter in output_list_counter]
    attention_score_matrix = calculate_attention_score_matrix(cluster_execution_outputs)

    # Calculate cluster pass rates
    test_inputs = task_items[0]["test_inputs"][:num_limit_test_cases]
    intersection_inputs, intersection_input_indices = [], []
    for i, test_input in enumerate(test_inputs):
        if test_input in model_generated_input_output_dict:
            intersection_inputs.append(test_input)
            intersection_input_indices.append(i)

    model_generated_outputs = [model_generated_input_output_dict[test_input] for test_input in intersection_inputs]

    pass_rates = []
    for outputs in cluster_execution_outputs:
        _outputs = [outputs[i] for i in intersection_input_indices]
        pass_rate = calculate_cluster_pass_rate(_outputs, model_generated_outputs) 
        pass_rates.append(pass_rate)

    num_solutions = [counter[1] for counter in output_list_counter]
    passed_results = [counter[2] for counter in output_list_counter]

    if count_invalid > 0:
        if len(attention_score_matrix.shape) == 2:
            n = attention_score_matrix.shape[0]
            new_matrix = np.zeros((n + 1, n + 1))
            new_matrix[:n,:n] = attention_score_matrix
            attention_score_matrix = new_matrix
        else:
            attention_score_matrix = np.array([0])
        num_solutions.append(0)
        pass_rates.append(0)
        passed_results.append(passed_results_of_invalid)

    if sum(num_solutions) == 0:
        correctness_features = (np.array(num_solutions), np.array(pass_rates))
    else:
        correctness_features = (np.array(num_solutions) / sum(num_solutions), np.array(pass_rates))

    return task_id, attention_score_matrix, correctness_features, passed_results
