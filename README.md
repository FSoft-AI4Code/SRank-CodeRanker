<div align="center">

# Functional Overlap Reranking for Neural Code Generation
[![arXiv](https://img.shields.io/badge/arXiv-2311.03366-b31b1b.svg)](https://arxiv.org/abs/2311.03366)

</div>

This repository contains the code implementation for the paper **[Functional Overlap Reranking for Neural Code Generation](https://arxiv.org/abs/2311.03366)**, accepted as a long paper to ACL Findings 2024.

Authors: Hung Q. To, Minh H. Nguyen, Nghi D. Q. Bui

## Introduction
We introduce **SRank**, a novel reranking strategy for selecting the best solutions from code generation models, focusing on modeling the relationships between clusters of solutions. By quantifying the functional overlap between solution clusters, our approach provides a superior ranking strategy for code solutions. Empirical results demonstrate that our method achieves remarkable improvements in the pass@1 score. For instance, on the Human-Eval benchmark, we achieve 69.66% with Codex002, 75.31% with WizardCoder, 53.99% with StarCoder, and 60.55% with CodeGen, significantly surpassing state-of-the-art code generation reranking methods like CodeT and Coder-Reviewer by an average margin of ≈6.1%. Compared to random sampling, we observe an average improvement of ≈23.07% on Human-Eval and 17.64% on MBPP, showcasing the robustness and superiority of our approach even in scenarios with limited test inputs.

## Main Results
The tables below show the `pass@1` results of **SRank** on various benchmarks in the zero-shot setting compared to baselines and state-of-the-art methods.

### HumanEval

|              | WizardCoder34B | WizardCoder15B | CodeGen2.5-Instruct | StarCoder | Codex002 | CodeGen16B |
|--------------|----------------|----------------|---------------------|-----------|----------|------------|
| Greedy       | 68.90          | 50.61          | 28.05               | 39.63     | 47.00    | 29.70      |
| CodeT        | 72.36          | 58.64          | 56.81               | 50.51     | 65.80    | 36.70      |
| Coder-Reviewer | -            | 49.37          | 45.63               | 38.71     | 66.90    | 42.60      |
| Random       | 59.88          | 45.20          | 26.68               | 32.55     | 37.06    | 22.78      |
| **SRank**        | **75.31**      | **59.99**      | **60.55**           | **53.99** | **69.66**| **43.07**   |

*Table 1: Results of pass@1 on HumanEval.*

### MBPP-S

|              | WizardCoder34B | WizardCoder15B | CodeGen2.5-Instruct | StarCoder | Codex002 | CodeGen16B |
|--------------|----------------|----------------|---------------------|-----------|----------|------------|
| Greedy       | 60.42          | 51.29          | 42.86               | 45.90     | 58.10    | 42.40      |
| CodeT        | 63.39          | 58.18          | 55.02               | 58.05     | 67.70    | 49.50      |
| Coder-Reviewer | -            | 52.52          | 52.74               | 49.48     | 64.70    | 50.30      |
| Random       | 54.37          | 45.72          | 34.60               | 39.26     | 47.50    | 31.54      |
| **SRank**        | **64.14**      | **59.01**      | **57.02**           | **58.38** | **69.25**| **51.03**   |

*Table 2: Results of pass@1 on MBPP-S.*

### APPS

| Method  | Introduction | Interview | Competition |
|---------|--------------|-----------|-------------|
| Random  | 20.35        | 3.11      | 0.74        |
| Greedy  | 27.20        | 5.10      | 1.80        |
| CodeT   | 34.60        | 8.10      | 2.20        |
| **SRank**   | **37.79**    | **9.53**  | **3.29**    |

*Table 3: Results of pass@1 on APPS benchmark using Codex002.*

Please refer to our paper for detailed explanations of these results and additional findings, including ablation studies.

## Installation
To set up the environment and dependencies, follow these steps:

1. Ensure you have Python 3.9.17 installed.
2. Install [pyminifier](https://github.com/liftoff/pyminifier/tree/master) from source. Note that you may need to revert setuptools to an older version: `pip install setuptools==57.5.0`. Refer to the [pyminifier issues](https://github.com/liftoff/pyminifier/issues) for potential fixes.
3. Install [`human-eval`](https://github.com/openai/human-eval) from source.
4. Install additional dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
This repository facilitates conducting experiments with the models and datasets listed in our paper.

### Available Models:
- wizardcoder34B
- wizardcoder15B
- codegen25
- starcoder
- davinci002
- codegen16B

### Available Datasets:
- humaneval
- mbpp
- apps

### Pipeline Overview:
Our CodeLLM-based code generation process involves three main steps:
1. **CodeLLM-based Generation**
    - Code solution generation
    - Test case generation
    - Post-processing code solutions and test cases
2. **Code Execution**
3. **Reranking**

### Variables Used in Scripts:
- `device_ids`: GPU device IDs
- `model`: Select one from the available models listed above
- `dataset`: Select one from the available datasets listed above
- `max_sequence_length`: Max sequence length for LLM
- `number_of_sequences`: Number of samples drawn from LLM
- `running_script`: Python script for the corresponding model
- `reranking_method`: Reranking method applied to code solution clusters (options: `random`, `srank`)

Default hyperparameters: `temperature=0.8`, `top_p=0.95`.

## Steps to Reproduce Results

### CodeLLM-based Generation
To generate code solutions, navigate to the appropriate directory and run the script:
```bash
cd generation/gen_code/sh
./run.sh ${device_ids} ${model} ${dataset} ${max_sequence_length} ${number_of_sequences} ${running_script}
```
Example:
```bash
./run.sh 0,1,2,3 wizardcoder humaneval 2048 8 wizardcoder.py
```
Post-process the raw data:
```bash
./postprocess.sh ${model} ${dataset}
```
Results are saved to `preds/${dataset}/${model}/postprocessed_T${temperature}_N${num_samples}.jsonl`.

### Test Case Generation
Navigate to the test case generation directory and run the script:
```bash
cd generation/gen_test/sh
./run.sh ${device_ids} ${model} ${dataset} ${max_sequence_length} ${number_of_sequences}
```
Example:
```bash
./run.sh 0,1,2,3 wizardcoder humaneval 2048 8 wizardcoder.py
```
Post-process the test cases:
```bash
./postprocess.sh ${model} ${dataset}
```
Results are saved to `preds/${dataset}/${model}/postprocessed_T${temperature}_N${num_samples}.jsonl`.

### Code Execution
Navigate to the execution directory and run the command:
```bash
cd execution/sh
./run.sh ${model} ${dataset}
```
Execution results are saved to `results/${dataset}/${model}/T${temperature}_N${num_samples}/`.

### Reranking
Navigate to the reranking directory and run the script:
```bash
cd reranking/sh
./run.sh ${model} ${dataset} ${temperature} ${num_samples} ${reranking_method}
```
Example:
```bash
./run.sh wizardcoder humaneval 0.8 100 srank
```

## Acknowledgement
This code base is adapted from:
- [WizardLM](https://github.com/nlpxucan/WizardLM)
- [CodeT](https://github.com/microsoft/CodeT)
- [Coder-Reviewer](https://github.com/facebookresearch/coder_reviewer_reranking)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions or collaborations, please contact:
- Hung Quoc To
- Email: tqh262@gmail.com
