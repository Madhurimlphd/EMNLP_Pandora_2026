import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification,get_linear_schedule_with_warmup,AutoModelForSequenceClassification,AutoTokenizer,XLMRobertaForSequenceClassification,RobertaTokenizer,RobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score,recall_score
import time

col_name = sys.argv[1]


file_path=sys.argv[2]
data = pd.read_csv(file_path)




print("*******************************************************************************************")
print("Received column name is:", col_name)
print("Received file path is:", file_path)
print("length of the dataset",len(data))




print("length of the updated dataset",len(data))

data = data.reset_index(drop=True)
# Get parameter from command line

data = data.dropna(subset=["tweet", col_name])
if data[col_name].isin(["YES", "NO"]).all():
    data[col_name] = data[col_name].map({"NO": 0, "YES": 1})


print("-------------------URL/lower processing ---------------------")

import re
def remove_urls_and_lower(text):
    # Define the regex pattern for URLs starting with http or https
    url_pattern = re.compile(r'http[s]?://\S+')
    # Substitute the URLs with an empty string
    cleaned_text = url_pattern.sub('', text)
    cleaned_text = cleaned_text.lower()
    return cleaned_text.strip()

tweet_processed=data["tweet"].apply(remove_urls_and_lower)


train_texts, val_texts, train_labels, val_labels = train_test_split(tweet_processed.values, data[col_name].values,test_size=0.2, random_state=42,)



print("-------------------Train and Test size ---------------------")
print(len(train_texts), len(train_labels))
print(len(val_texts), len(val_labels))

print("-------------------Set seed for reproducibility ---------------------")
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


print("------------------Tokenizer and Model setup-------------------------")

#print("Roberta-base")
model_path = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path,num_labels=2)  







print("-------------Setting dataset---------")
class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Define dataset
max_len = 128
train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, max_len)
val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer, max_len)

print("----------------------DataLoader ------------------------------")
# Create DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


print("-----------------------Device setup--------------------------")
# Model and Device Setup




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("-----GPU Status----")
if "cuda" in str(model.device):
    print("Model running on GPU")
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("Model is running on CPU")

print("CUDA available:", torch.cuda.is_available())
print("Model device:", model.device)
print("Model device Parameters:", next(model.parameters()).device)



# Optimizer
print("--------------Optimizer and scheduler------------")
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 3
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)



print("------------------------Training Loop-------------------------")


for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, leave=True)
    t0 = time.perf_counter()
    
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        
        #Gradient Clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()    #updates LR scheduler
        
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())
    epoch_time = time.perf_counter() - t0
    print(f"Epoch {epoch} Loss: {total_loss / len(train_loader)} Time:{epoch_time}")
    
#Evaluation
# Evaluation
model.eval()
correct = 0
total = 0
all_predictions=[]
all_labels=[]
t_eval0 = time.perf_counter()

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        all_predictions.extend(predictions.cpu().numpy())  
        all_labels.extend(labels.cpu().numpy())        

eval_time = time.perf_counter() - t_eval0
accuracy = correct / total





print("------------------------Results-------------------------")
print("::::Results for:::: ", col_name)
print("::::file name::::::",file_path)
print(f"Validation Accuracy: {accuracy:.4f} Eval time :{eval_time}")


f1_macro = f1_score(all_labels, all_predictions, average='macro') 

pre=precision_score(all_labels, all_predictions, average='macro')
recall=recall_score(all_labels, all_predictions, average='macro')

print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"Precision (Macro): {pre:.4f}")
print(f"Recall (Macro): {recall:.4f}")






