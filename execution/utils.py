import re

def convert_test_case_to_input_output(test_case, prompt, entry_point):
    test_case = re.sub("'''.*?'''", "", test_case, flags=re.DOTALL)
    test_case = re.sub('""".*?"""', "", test_case, flags=re.DOTALL)

    test_case = test_case.replace("\n", "")

    if "," in test_case:
        comma_seperated = test_case.split(",")
        message = comma_seperated[-1]
        try:
            message = eval(message)
            test_case = ",".join(comma_seperated[:-1])
        except Exception as e:
            pass

    comparison = test_case.lstrip("assert").strip()
    if comparison.startswith("(") and comparison.endswith(")"):
        comparison = comparison[1:-1]
        test_case = "assert {}".format(comparison)

    if "==" in test_case:
        match = re.match(r'assert\s+(.*?)\s+(==)\s+(.+)', test_case)
    else:
        match = re.match(r'assert\s+(.*?)\s+(is)\s+(True|False|None)', test_case)
    if match:
        input_str, operator, output_str = match.groups()

        if re.search("{}\(.*?\)".format(entry_point), output_str):
            input_str, output_str = output_str, input_str

        if not re.search("{}\(.*?\)".format(entry_point), input_str):
            return "", ""

        if not check_test_input_validation(prompt, input_str):
            print(f"Invalid input:\t{test_case = }\t{input_str = }")
            return "", ""

        return input_str, output_str
    return "", ""

def check_test_input_validation(prompt, test_input):
    if len(test_input.strip()) == 0:
        return False
    try:
        program = prompt + f"\n    pass\n{test_input}"
        compile(program, "", "exec")
        return True
    except:
        return False
