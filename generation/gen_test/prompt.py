def generate_prompt(input, entry_point, model_name, for_decoder=True):
    if not input.endswith("\n"):
        input += "\n"
    if model_name.startswith("bigcode/starcoder"):
        INSTRUCTION = f"<filename>solutions/solution_1.py\n{input}    pass\n\n# check the correctness of {entry_point}\nassert {entry_point}("
    elif model_name.startswith("meta-llama/Llama-2") or model_name.startswith("codellama/CodeLlama-34b-Python-hf"):
        INSTRUCTION = f"{input}    pass\n\n# check the correctness of {entry_point}\nassert {entry_point}("
#     elif model_name.startswith("WizardLM/WizardCoder") or model_name.startswith("Salesforce/codegen25"):
#         INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
# 
# ### Instruction:
# I have this function stub, please generate 50 test cases for this function. The function stub is as follow:
# ```python
# {input}    pass
# ```
# - Each test case is in the form of assertion statement, for example: assert {entry_point}(...) == ...
# - Each test case is in a single line
# - The length of each test case should be too long, ideally less than or equal to 150 letters
# - The test input should not be too long
# - The inputs of test cases should be diverse and cover corner cases of the function
# - Test cases should not be repeated
# 
# ### Response: Here are 50 test cases for function `{entry_point}`:
# assert {entry_point}("""
    else:
    # elif model_name.startswith("Salesforce/instructcodet5p"):
        INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
    I have this function stub, please generate 50 test cases for this function. The function stub is as follow:
    ```python
    {input}    pass
    ```
    - Each test case is in the form of assertion statement, for example: assert {entry_point}(...) == ...
    - Each test case is in a single line
    - The length of each test case should be too long, ideally less than or equal to 150 letters
    - The test input should not be too long
    - The inputs of test cases should be diverse and cover corner cases of the function
    - Test cases should not be repeated

### Response:"""

        if for_decoder:
            INSTRUCTION += f"Here are 50 test cases for function `{entry_point}`:\nassert {entry_point}("

    return INSTRUCTION
