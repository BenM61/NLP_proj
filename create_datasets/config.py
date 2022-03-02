from pathlib import Path
import os.path

# path of the root folder
ROOT_PATH = Path(os.path.realpath(__file__)).parent

SONGS_PATH = os.path.join(ROOT_PATH, "SONGS")

# files for the metadata
OVERALL_STATS_PATH = os.path.join(ROOT_PATH, "stats.txt")
TITLES_GENRES_PATH = os.path.join(ROOT_PATH, "titles.pickle")

TRAIN_TEST_SPLIT = 0.8