##In this notebook, we used hugging face Mistral model to annotate the EXIST tweets
##mistralai/Mistral-7B-Instruct-v0.3

#https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

import os
import torch

from huggingface_hub import login

login(token="")

from transformers import AutoTokenizer, AutoModelForCausalLM



print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Calling model")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
#model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",torch_dtype=torch.float16,device_map="auto") 
#model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

print("using model.to(device)")
print("Model device:", model.device)
print("torch available:",torch.cuda.is_available())
print("memory used:",torch.cuda.memory_allocated()/1024**2, "MB")

if "cuda" in str(model.device):
    print("Model is running on GPU")
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("Model is running on CPU")

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
#print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
#print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:],skip_special_tokens=True).strip())

    #print("torch cuda memory",torch.cuda.memory_allocated() / 1024**3, "GB allocated")
    #print("torch cuda memory",torch.cuda.memory_reserved() / 1024**3, "GB reserved")
    
    label = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    return label

#labels = [classify_text(t) for t in texts]

#for text, label in zip(texts, labels):
#    print(label, "-", text)

df["mistral_anno"] = df["tweet"].apply(classify_text)

df.to_csv("LLM_annotated_dataset/mistral_EXIST2023.csv", index=False)