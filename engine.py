import torch
import utils
import numpy
import torch.nn as nn


def loss(o1, t1, o2, t2):
    loss_1 = nn.BCEWithLogitsLoss()(o1, t1)
    loss_2 = nn.BCEWithLogitsLoss()(o2, t2)
    return loss_1 + loss_2


def train(data_loader, model, optimizer, scheduler, device):
    model.train()

    for ix, batchx in enumerate(data_loader):

        ids = batchx['ids']
        masks = batchx['attention_mask']
        t_ids = batchx['token_type_ids']
        start = batchx['start_vector']
        end = batchx['end_vector']

        ids = ids.to(device, dtype=torch.long)
        masks = masks.to(device, dtype=torch.long)
        t_ids = t_ids.to(device, dtype=torch.long)
        start = start.to(device, dtype=torch.long)
        end = end.to(device, dtype=torch.long)

        optimizer.zero_grad()
        o1, o2 = model(ids=ids, masks=masks, token_type_ids=t_ids)

        _loss = loss(o1, start, o2, end)
        _loss.backward()
        optimizer.step()
        scheduler.step()


def eval(data_loader, model, device):
    model.eval()

    fin_output_start = []
    fin_output_end = []
    fin_pad_len = []
    fin_tweet_tokens = []
    fin_sentiment = []
    fin_selected_text = []
    fin_tweet = []

    start_idx, end_idx = [], []
    for ix, batchx in enumerate(data_loader):
        ids = batchx['ids']
        masks = batchx['attention_mask']
        t_ids = batchx['token_type_ids']
        start = batchx['start_vector']
        end = batchx['end_vector']
        tweet = batchx['tweet']
        sentiment = batchx['sentiment']
        selected_text = batchx['selected_text']
        pad_length = batchx['pad_length']
        tweet_tokens = batchx['tweet_tokens']

        ids = ids.to(device, dtype=torch.long)
        masks = masks.to(device, dtype=torch.long)
        t_ids = t_ids.to(device, dtype=torch.long)

        o1, o2 = model(ids=ids, masks=masks, token_type_ids=t_ids)

        fin_output_start.append(torch.sigmoid(o1).cpu().detach().numpy())
        fin_output_end.append(torch.sigmoid(o2).cpu().detach().numpy())
        fin_pad_len.extend(torch.sigmoid(pad_length).cpu().detach().numpy().tolist())
        fin_tweet.extend(tweet)
        fin_selected_text.extend(selected_text)
        fin_sentiment.extend(sentiment)
        fin_tweet_tokens.extend(tweet_tokens)

    fin_output_start = numpy.vstack(fin_output_start)
    fin_output_end = numpy.vstack(fin_output_end)

