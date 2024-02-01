import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model(
    load_8bit: bool = False,
    base_model: str = "bigcode/starcoder",
    device: str = "cpu",
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"
    )

    if base_model == "Salesforce/codegen25-7b-instruct":
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    model.config.pad_token_id = tokenizer.pad_token_id

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    return tokenizer, model
