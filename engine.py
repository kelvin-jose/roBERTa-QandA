import torch
import utils
import numpy as np
import torch.nn as nn


def calculate_jaccard_score(
        original_tweet,
        target_string,
        sentiment_val,
        idx_start,
        idx_end,
        offsets,
        verbose=False):
    if idx_end < idx_start:
        idx_end = idx_start

    filtered_output = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            filtered_output += " "

    if len(original_tweet.split()) < 2:
        filtered_output = original_tweet

    jac = utils.jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output


def loss(start_logits, end_logits, start_positions, end_positions):
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss


def train(data_loader, model, optimizer, scheduler, device, fold, epoch):
    model.train()

    for ix, batchx in enumerate(data_loader):
        ids = batchx['ids']
        masks = batchx['attention_mask']
        t_ids = batchx['token_type_ids']
        start_idx = batchx['start_index']
        end_idx = batchx['end_index']
        ids = ids.to(device, dtype=torch.long)
        masks = masks.to(device, dtype=torch.long)
        t_ids = t_ids.to(device, dtype=torch.long)
        start = start_idx.to(device, dtype=torch.long)
        end = end_idx.to(device, dtype=torch.long)

        optimizer.zero_grad()
        o1, o2 = model(ids=ids, masks=masks, token_type_ids=t_ids)

        _loss = loss(o1, o2, start, end)
        _loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"Fold: {fold}  Epoch: {epoch}  Batch: {ix}   Train loss: {_loss}")


def eval(data_loader, model, device, fold):
    model.eval()

    for ix, batchx in enumerate(data_loader):
        print(f"Fold: {fold}  Batch: {ix}")
        ids = batchx['ids']
        masks = batchx['attention_mask']
        t_ids = batchx['token_type_ids']
        start = batchx['start_index']
        end = batchx['end_index']
        offsets = batchx['tweet_offsets']
        sentiment = batchx['sentiment']
        selected_text = batchx['selected_text']
        tweets = batchx['tweet']

        ids = ids.to(device, dtype=torch.long)
        masks = masks.to(device, dtype=torch.long)
        t_ids = t_ids.to(device, dtype=torch.long)

        o1, o2 = model(ids=ids, masks=masks, token_type_ids=t_ids)
        _loss = loss(o1, o2, start, end)

        outputs_start = torch.softmax(o1, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(o2, dim=1).cpu().detach().numpy()

        jaccard_scores = []
        for px, tweet in enumerate(tweets):
            selected_tweet = selected_text[px]
            tweet_sentiment = sentiment[px]
            jaccard_score, _ = calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                idx_start=np.argmax(outputs_start[px, :]),
                idx_end=np.argmax(outputs_end[px, :]),
                offsets=offsets[px]
            )
            jaccard_scores.append(jaccard_score)

    return np.mean(jaccard_scores)

