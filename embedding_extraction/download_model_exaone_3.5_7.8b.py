# load the model, which will automatically download the model if it is not already in the cache directory

import transformers
print(transformers.__version__)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

# the model automatically saves the model in the cache directory
# save_directory = "./model/exaone_model_3.5"
