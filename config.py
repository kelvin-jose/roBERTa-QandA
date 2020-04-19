from tokenizers import ByteLevelBPETokenizer
from os.path import join

MAX_LEN = 192
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 10
BERT_PATH = '../input/bert-base-uncased/'
ROBERTA_PATH = '../input/robertabase/'
ROBERTA_CONFIG = join(ROBERTA_PATH, 'roberta-base-config.json')
MODEL_PATH = 'model.bin'
TRAIN_DATA = '../input/train.csv'
TOKENIZER = ByteLevelBPETokenizer(join(ROBERTA_PATH, 'roberta-base-vocab.json'),
                                  join(ROBERTA_PATH, 'roberta-base-merges.txt'),
                                  lowercase=True,
                                  add_prefix_space=True)
