from create_db import config_db
import create_db.file_utils as utils
import config_dataset as config
from LyricsDataset import LyricsDataset
import preprocessing

import os
import random
import shutil

def delete_dataset_files():
  print(f"[DEBUG] Deleting dataset files...")
  shutil.rmtree(config.SONGS_PATH, ignore_errors=True)
  os.remove(config.OVERALL_STATS_PATH)
  os.remove(config.TITLES_GENRES_PATH)

# returns whether if the song *after* preprocess is valid, and the target path
# if True- also adds the preprocessed file to the right path
# if False- does nothing (the path returned is invalid)
# needs to load the dicts before and save them after!
def copy_then_preprocess_song(from_path, target_root, 
                              DB_title_dict, datasets_title_dict):
  from_path = str(from_path)
  filename = from_path.split(os.path.sep)[-1]
  title = filename[:-4]
  to_path = os.path.join(target_root, filename)

  shutil.copy(from_path, to_path)
  preprocessing.preprocess(to_path)

  # TODO: add checks to varify the preproccesd file is valid
  # (open every time)
  valid = True
  with open(to_path) as f:
    if not utils.is_song_file_valid(f):
      valid = False

  with open(to_path) as f:
    if not preprocessing.no_chorus_no_repeat(f):
      valid = False
  
  if valid:
    datasets_title_dict[title] = DB_title_dict[title]
  else:
    os.remove(to_path)

  return valid, to_path

def create_datasets():
  print(f"[INFO] Starts creating Datasets...")
  if os.path.exists(config.TITLES_GENRES_PATH):
    print(f"[INFO] Loading saved files for datasets...")
    datasets_title_dict = utils.load_dict(config.TITLES_GENRES_PATH)

    paths = [os.path.join(config.SONGS_PATH, title + ".txt") 
              for title in datasets_title_dict.keys()]
  else:
    print(f"[INFO] Creating files for datasets...")
    paths = []
    DB_root = config_db.DB_PATH

    DB_title_dict = utils.load_dict(config_db.TITLES_GENRES_PATH)
    datasets_title_dict = utils.load_dict(config.TITLES_GENRES_PATH)

    titles = list(DB_title_dict.keys())
    
    #print(f"[DEBUG] Copying and preprocessing song files...")
    os.mkdir(config.SONGS_PATH)
    for title in titles:
      # get a path to this song
      label = DB_title_dict[title][0]
      song_path = os.path.join(DB_root, label, title + ".txt")

      valid, path = copy_then_preprocess_song(song_path, config.SONGS_PATH, 
                        DB_title_dict, datasets_title_dict)
      if valid:
        paths.append(path)
      
    utils.save_dict(datasets_title_dict, config.TITLES_GENRES_PATH)
    #print(f"[DEBUG] Updating stats...")
    utils.save_datasets_stats(config.OVERALL_STATS_PATH, datasets_title_dict)

  random.Random(666).shuffle(paths)
  split_ind = int(len(paths) * config.TRAIN_TEST_SPLIT)
  train_paths = paths[:split_ind]
  test_paths = paths[split_ind:]

  # make the dataset objects
  train_dataset = LyricsDataset(datasets_title_dict, train_paths)
  test_dataset = LyricsDataset(datasets_title_dict, test_paths)

  return train_dataset, test_dataset


#delete_dataset_files()

# dataloader example (we need shuffle to true only on train)

from torch.utils.data import DataLoader
t, tt = create_datasets()
td = DataLoader(t, 4, False)
for i in next(iter(td)):
  print(i)