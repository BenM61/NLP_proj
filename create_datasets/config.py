from pathlib import Path
import os.path

# path of the root folder
ROOT_PATH = Path(os.path.realpath(__file__)).parent

TRAIN_PATH = os.path.join(ROOT_PATH, "TRAIN")

TEST_PATH = os.path.join(ROOT_PATH, "TEST")

TRAIN_TEST_SPLIT = 0.8