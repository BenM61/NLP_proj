from pathlib import Path
import os.path
from transformers import TOKENIZER_MAPPING, T5Tokenizer

# path of the root folder
ROOT_PATH = Path(os.path.realpath(__file__)).parent

SONGS_PATH = os.path.join(ROOT_PATH, "SONGS")

# files for the metadata
OVERALL_STATS_PATH = os.path.join(ROOT_PATH, "stats.txt")
TITLES_GENRES_PATH = os.path.join(ROOT_PATH, "titles.pickle")

# tokenizer parameters
TOKENIZER = T5Tokenizer.from_pretrained("t5-base")
TOKENIZER_SOURCE_MAX_LENGTH = 1200
TOKENIZER_TARGET_MAX_LENGTH = 30

TRAIN_TEST_SPLIT = 0.8