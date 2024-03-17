
<div align="center">


# Neural Rankers for Code Generation via Inter-Cluster Modeling
[![arXiv](https://img.shields.io/badge/arXiv-2305.06156-b31b1b.svg)](https://arxiv.org/abs/2311.03366)

</div>


## Introduction

We introduce SRank, a novel reranking strategy for selecting the best solution from code generation that focuses on modeling inter-cluster relationship. By quantifying the functional overlap between clusters, our approach provides a better ranking strategy of code solutions. Empirical results show that our method achieves a remarkable results on pass@1 score. For instance, on the Human-Eval benchmark, we achieve 69.66% in pass@1 with Codex002, 75.31% for WizardCoder, 53.99% for StarCoder and 60.55% for CodeGen, which surpass the state-of-the-arts solution ranking methods, such as CodeT and Coder-Reviewer on the same CodeLLM with significant margin (≈6.1% improvement on average). Comparing to the random sampling method, we can achieve an average improvement of ≈23.07% on Human-Eval and 17.64\% on MBPP. Even in scenarios with limited test inputs, our approach demonstrates robustness and superiority, marking a new state-of-the-arts in code generation reranking.

## Installation
1. All experiments are run with `python==3.9.17`. 
2. Install [pyminifier](https://github.com/liftoff/pyminifier/tree/master) from source.
Installing `pyminifier` requires reverting setup tools to an older version (`pip install setuptools==57.5.0`). 
For other issues of installing `pyminifier`, checkout their [issues](https://github.com/liftoff/pyminifier/issues) for potential fixes.
3. Install [`human-eval`](https://github.com/openai/human-eval) from source.
4. Install the other packages by 
```bash
pip install -r requirements.txt
```
## Usage
Available models:
- wizardcoder34B
- wizardcoder15B
- codegen25
- starcoder
- davinci002
- codegen16B

Available datasets:
- humaneval
- mbpp
- apps

### Data
The processed results will be saved at these locations with pre-defined file names
- Post-processed code solutions: `generation/gen_code/preds/${dataset}/${model}/postprocessed_T${temperature}_N${num_samples}.jsonl`
- Post-processed test cases: `generation/gen_test/preds/${dataset}/${model}/postprocessed_T${temperature}_N${num_samples}.jsonl`
- Execution results: `execution/results/${dataset}/${model}/T${temperature}_N{$num_samples}/*`

## Data generation
### Generating code solutions
```bash
cd generation/gen_code/sh
./run.py ${device_ids} ${model} ${dataset} ${max_sequence_length} ${number_of_sequences} ${running_script}
```
For example, running `wizardcoder` on `humaneval`
```bash
cd generation/gen_code/sh
./run.sh 0,1,2,3 wizardcoder humaneval 2048 8 wizardcoder.py
```
Results are saved to `generation/gen_code/preds/${dataset}/${model}/T${temperature}_N${num_samples}/`

### 1st step: Generating test cases
```bash
cd generation/gen_test/sh
./run.py ${device_ids} ${model} ${dataset} ${max_sequence_length} ${number_of_sequences} ${running_script}
```
For example, running `wizardcoder` on `humaneval`
```bash
cd generation/gen_test/sh
./run.sh 0,1,2,3 wizardcoder humaneval 2048 8 wizardcoder.py
```

Results are saved to `generation/gen_test/preds/${dataset}/${model}/T${temperature}_N${num_samples}/`

## 2nd step: Post-processing raw generation
### Post-processing code solutions
```bash
cd generation/gen_code/sh
./postprocess.sh ${model} ${dataset}
```
Results are saved to `generation/gen_code/preds/${dataset}/${model}/postprocessed_T${temperature}_N${num_samples}.jsonl`

### Post-processing test cases
```bash
cd generation/gen_test/sh
./postprocess.sh ${model} ${dataset}
```
Results are saved to `generation/gen_test/preds/${dataset}/${model}/postprocessed_T${temperature}_N${num_samples}.jsonl`

## 3rd step. Execution
```bash
cd execution/sh
./run.sh ${model} ${dataset}
```

Execution results are saved to `execution/results/${dataset}/${model}/T${temperature}_N{$num_samples}/`. The folder contains the following files:
- `ground_truth_exec_result.pkl`: Execution results of code solutions on ground truth test cases, as provided by benchmark datasets.
- `model_generated_test_cases.pkl`: Processed model-generated test cases, excluding those with syntactic and partially semantic inaccuracies.
- `test_inputs_exec_result.pkl`: Execution outputs of code solutions on model-generated test cases.

## 4th step: Reranking
Available reranking methods:
- attention
- random

```bash
./run.sh ${model} ${dataset} ${temperature} ${num_samples} ${reranking_method}
```
For example, running reranking `wizardcoder15B` on `humaneval`
```bash
cd reranking/sh
./run.sh wizardcoder humaneval 0.8 100 attention
```

# Acknowledgement
This code base is adapted from
- [WizardLM](https://github.com/nlpxucan/WizardLM)
- [CodeT](https://github.com/microsoft/CodeT)
- [Coder-Reviewer](https://github.com/facebookresearch/coder_reviewer_reranking)
