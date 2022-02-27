from pathlib import Path
import os

################################# DB parameters #################################
# path of the root folder
ROOT_PATH = Path(os.path.realpath(__file__)).parent

DB_PATH = os.path.join(ROOT_PATH, "DB")

# files for the metadata
OVERALL_STATS_PATH = os.path.join(DB_PATH, "stats.txt")
TITLES_GENRES_PATH = os.path.join(DB_PATH, "titles.pickle")

# temp files to set the metadata 
TEMP_TITLES_GENRES_PATH = os.path.join(DB_PATH, "temp_titles.pickle")

# folders for the song's lyrics to be in
ROCK_PATH = os.path.join(DB_PATH, "ROCK")
POP_PATH = os.path.join(DB_PATH, "POP")
HIP_HOP_PATH = os.path.join(DB_PATH, "HIP_HOP")
ELECTRONIC_PATH = os.path.join(DB_PATH, "ELECTRONIC")
BLUES_PATH = os.path.join(DB_PATH, "BLUES")
JAZZ_PATH = os.path.join(DB_PATH, "JAZZ")
FUNK_PATH = os.path.join(DB_PATH, "FUNK")
#  TODO:  and so on for the other categories...


LABELS = ["POP", "HIP_HOP", "ROCK", "ELECTRONIC", "BLUES", "JAZZ", "FUNK"]
#  TODO:  and so on for the other categories...

GENRE_TO_LABEL = {"Pop" : "POP", "Hip%20Hop" : "HIP_HOP", "Rock" : "ROCK", "Funk%20--%20Soul" : "FUNK",
"Electronic" : "ELECTRONIC", "Blues" : "BLUES", "Jazz" : "JAZZ"}
#  TODO:  and so on for the other categories...

LABEL_TO_PATH = {"POP" : POP_PATH, "HIP_HOP" : HIP_HOP_PATH, "ROCK" : ROCK_PATH, "FUNK" : FUNK_PATH,
"ELECTRONIC" : ELECTRONIC_PATH, "BLUES" : BLUES_PATH, "JAZZ" : JAZZ_PATH}
#  TODO:  and so on for the other categories...

################################# dataset parameters #################################
TRAIN_VAL_SPLIT = 0.8