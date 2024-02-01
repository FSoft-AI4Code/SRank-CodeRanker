import zlib
from nltk.translate.bleu_score import sentence_bleu
from pyminifier_canonicalize import clean_comment

def check_validity(item):
    if not item["test_inputs_passed"]:
        return False
    if len(item["outputs"]) > 1 and all(output[1] == item["outputs"][0][1] for output in item["outputs"]):
        return False

    return True

def filter_empty(x, remove_function_header=False):
    code = x["completion"]
    if remove_function_header:
        code = "\n".join(
            [l for l in code.split("\n") if not l.strip().startswith("def")]
        )
    try:
        code = clean_comment(code)
    except:
        code = ""
    return code.strip() not in ["", "pass", "return"]


def filter_repeat(x, threshold=0.25):

    bytes_x = bytes(x, encoding="utf-8")
    comp_x = zlib.compress(bytes_x)
    return len(comp_x) / len(bytes_x) > threshold
