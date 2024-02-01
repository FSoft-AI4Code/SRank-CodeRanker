def generate_prompt(input, model_name):
    if model_name.startswith("bigcode/starcoder"):
        INSTRUCTION = f"<filename>solutions/solution_1.py\n# Here is the correct implementation of the code exercise\n{input}"
    elif model_name.startswith("meta-llama/Llama-2"):
        INSTRUCTION = f"# Here is the correct implementation of the code exercise\n{input}"
    elif model_name.startswith("WizardLM/WizardCoder") or model_name.startswith("Salesforce/codegen25") or model_name.startswith("Salesforce/instructcodet5p"):
        INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a Python script for this problem:
{input}

### Response:"""

    return INSTRUCTION
