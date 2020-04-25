import config
import torch


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
        tweet = ' ' + str(self.tweet[item])
        selected_text = ' ' + str(self.selected_text[item])
        sentiment = ' ' + str(self.sentiment[item])

        idx0, idx1 = -1, -1
        try:
            idx0 = str(self.tweet[item]).index(str(self.selected_text[item])) + 1
            idx1 = idx0 + len(selected_text) - 2
        except ValueError:
            pass

        char_vector = [0] * len(tweet)
        if idx0 + idx1 != -2:
            for ix in range(idx0, idx1 + 1):
                char_vector[ix] = 1

        tweet_ids = self.tokenizer.encode(tweet).ids
        tweet_offsets = self.tokenizer.encode(tweet).offsets

        target = []
        for ix, (offset1, offset2) in enumerate(tweet_offsets):
            if sum(char_vector[offset1: offset2]) > 0:
                target.append(ix)

        sentiment_encoded = {
            ' positive': 1313,
            ' negative': 2430,
            ' neutral': 7974
        }

        start_idx, end_idx = target[0] + 4, target[-1] + 4
        ids = [0] + [sentiment_encoded[sentiment]] + [2, 2] + tweet_ids + [2]
        token_type_ids = [0] * len(ids)
        masks = [1] * len(token_type_ids)
        tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]

        pad_length = self.max_len - len(ids)

        ids += [1] * pad_length
        token_type_ids += [0] * pad_length
        masks += [0] * pad_length
        tweet_offsets += [(0, 0)] * pad_length

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(masks, dtype=torch.long),
            'start_index': torch.tensor(start_idx, dtype=torch.long),
            'end_index': torch.tensor(end_idx, dtype=torch.long),
            'tweet_offsets': torch.tensor(tweet_offsets, dtype=torch.long),
            'tweet': tweet,
            'sentiment': sentiment,
            'selected_text': selected_text
        }

# import pandas as pd
# train = pd.read_csv('sample_csv.csv')
# td = TweetDataset(train['tweet'], train['selected_text'], train['sentiment'])
# print(td[0])



