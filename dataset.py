import config
import torch
import numpy


class TweetDataset:
    def __init__(self, tweet, selected_text, sentiment):
        self.tweet = tweet
        self.selected_text = selected_text
        self.sentiment = sentiment
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        tweet = " " + " ".join(str(self.tweet[item]).split())
        selected_text = " " + " ".join(str(self.selected_text[item]).split())

        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind: ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        tok_tweet = self.tokenizer.encode(tweet)
        input_ids_orig = tok_tweet.ids
        tweet_offsets = tok_tweet.offsets

        target_idx = []
        for j, (offset1, offset2) in enumerate(tweet_offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)

        targets_start = target_idx[0]
        targets_end = target_idx[-1]

        sentiment_id = {
            'positive': 1313,
            'negative': 2430,
            'neutral': 7974
        }

        input_ids = [0] + [sentiment_id[self.sentiment[item]]] + [2] + [2] + input_ids_orig + [2]
        token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
        mask = [1] * len(token_type_ids)
        tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
        targets_start += 4
        targets_end += 4

        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([1] * padding_length)
            mask = mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

        return {
            'ids': input_ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'targets_start': targets_start,
            'targets_end': targets_end,
            'orig_tweet': tweet,
            'orig_selected': selected_text,
            'sentiment': self.sentiment[item],
            'offsets': tweet_offsets
        }
        # tweet = ' ' + self.tweet[item]
        # selected_text = ' ' + self.selected_text[item]
        # sentiment = ' ' + self.sentiment[item]
        #
        # idx0, idx1 = -1, -1
        # try:
        #     idx0 = self.tweet[item].index(self.selected_text[item]) + 1
        #     idx1 = idx0 + len(selected_text) - 2
        # except ValueError:
        #     pass
        #
        # char_vector = [0] * len(tweet)
        # print(len(char_vector))
        # if idx0 + idx1 != -2:
        #     for ix in range(idx0, idx1 + 1):
        #         char_vector[ix] = 1
        #
        # tweet_ids = self.tokenizer.encode(tweet).ids
        # tweet_offsets = self.tokenizer.encode(tweet).offsets
        #
        # target = []
        # for ix, (offset1, offset2) in enumerate(tweet_offsets):
        #     if sum(char_vector[offset1: offset2]) > 0:
        #         target.append(ix)
        #
        # sentiment_encoded = {
        #     ' positive': 1313,
        #     ' negative': 2430,
        #     ' neutral': 7974
        # }
        #
        # start_idx, end_idx = target[0] + 4, target[-1] + 4
        # ids = [0] + [sentiment_encoded[sentiment]] + (2 * [2]) + tweet_ids + [2]
        # token_type_ids = [0, 0, 0, 0] + [0] * len(ids) + [0]
        # masks = [1] * len(token_type_ids)
        # tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
        #
        # pad_length = self.max_len - len(ids)
        #
        # token_type_ids += [0] * pad_length
        # ids += [1] * pad_length
        # masks += [0] * pad_length
        # tweet_offsets += [(0, 0)] * pad_length
        #
        # return {
        #     'ids': torch.tensor(ids, dtype=torch.long),
        #     'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        #     'attention_mask': torch.tensor(masks, dtype=torch.long),
        #     'start_index': torch.tensor(start_idx),
        #     'end_index': torch.tensor(end_idx),
        #     'tweet_offsets': tweet_offsets,
        #     'tweet': self.tweet[item],
        #     'sentiment': self.sentiment[item],
        #     'selected_text': self.selected_text[item]
        # }

import pandas as pd
train = pd.read_csv('train.csv')
td = TweetDataset(train['tweet'], train['selected_text'], train['sentiment'])
print(td[0])



