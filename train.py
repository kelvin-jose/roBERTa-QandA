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
from sklearn.model_selection import KFold
from transformers import AdamW, RobertaConfig
from transformers import get_linear_schedule_with_warmup


def main(fold, df_train, df_valid):

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
        tweet=df_valid.text.values,
        selected_text=df_valid.selected_text.values,
        sentiment=df_valid.sentiment.values
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

    model_config = RobertaConfig.from_pretrained(config.ROBERTA_CONFIG)
    model = Roberta(model_config)
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
        engine.train(train_data_loader, model, optimizer, scheduler, device, fold, epoch)
        current_jaccard = engine.eval(valid_data_loader, model, device, fold)
        print(f"Jaccard Score = {current_jaccard}")
        if current_jaccard > best_jaccard:
            torch.save(model.state_dict(), f"{fold}_{config.MODEL_PATH}")
            best_jaccard = current_jaccard


dfx = pd.read_csv(config.TRAIN_DATA)
kfold = KFold(5, True, 1)

for idx, (train, test) in enumerate(kfold.split(dfx)):

    df_train = dfx.iloc[train]
    df_test = dfx.iloc[test]
    main(idx, df_train, df_test)