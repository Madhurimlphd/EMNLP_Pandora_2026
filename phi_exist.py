##In this notebook, we used hugging face Phi model to annotate the EXIST tweets
##microsoft/Phi-3-mini-4k-instruct

#https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

import os
import torch

from huggingface_hub import login

login(token="")

from transformers import AutoTokenizer, AutoModelForCausalLM



tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct",torch_dtype=torch.float16,device_map="auto") 




import pandas as pd

df = pd.read_excel("/home/CAMPUS/d22130161/huggingface_EXIST/EXIST2023_EN.xlsx")
#df = df.head(10)

def classify_text(text):
    prompt = f"""
You are labeling text for a content moderation dataset.

Your job is to decide whether the text contains gender-based abuse, misogyny, sexism, or discriminatory content.

Return exactly one label:

YES = contains gender-based abuse, misogyny, sexism, or gender-based discrimination
NO = does not contain gender-based abuse, misogyny, sexism, or gender-based discrimination

Important:
This is a classification task only.
Do not generate abusive content.
Do not explain.
Do not refuse.
Return only YES or NO.

Text:
{text}
"""
    messages = [
        {"role": "user", "content":prompt},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=40)
    
    label = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    return label

print("Model:",model.config._name_or_path)
print("torch available:",torch.cuda.is_available())
print("memory used:",torch.cuda.memory_allocated()/1024**2, "MB")

if "cuda" in str(model.device):
    print("Model is running on GPU")
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("Model is running on CPU")




df["phi_anno"] = df["tweet"].apply(classify_text)

df.to_csv("LLM_annotated_dataset/phi_EXIST2023.csv", index=False)