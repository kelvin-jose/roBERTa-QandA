import config
import torch
import engine
import dataset
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import model_selection
from model import Roberta
from utils import jaccard
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

dfx = pd.read_csv(config.TRAIN_DATA)

df_train, df_valid = model_selection.train_test_split(
    dfx,
    test_size=0.1,
    random_state=42,
    stratify=dfx.sentiment.values
)

df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)

train_dataset = dataset.TweetDataset(
    tweet=df_train.text.values,
    selected_text=df_train.selected_text.values,
    sentiment=df_train.sentiment.values
)

train_data_loader = DataLoader(
    train_dataset,
    batch_size=config.TRAIN_BATCH_SIZE,
    num_workers=4
)

valid_dataset = dataset.TweetDataset(
    tweet=df_train.text.values,
    selected_text=df_train.selected_text.values,
    sentiment=df_train.sentiment.values
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=config.VALID_BATCH_SIZE,
    num_workers=1
)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = Roberta(config.ROBERTA_CONFIG)
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]

num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
optimizer = AdamW(optimizer_parameters, lr=3e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_steps
)

model = nn.DataParallel(model)

best_jaccard = 0
for epoch in range(config.EPOCHS):
    print('======== Epoch {} ========'.format(epoch))
    engine.train(train_data_loader, model, optimizer, scheduler, device)
    outputs, targets = engine.eval(valid_data_loader, model, device)
    outputs = np.array(outputs) >= 0.5
    jaccard = jaccard(targets, outputs)
    print(f"Accuracy Score = {accuracy}")
    if jaccard > best_jaccard:
        torch.save(model.state_dict(), config.MODEL_PATH)
        best_accuracy = accuracy