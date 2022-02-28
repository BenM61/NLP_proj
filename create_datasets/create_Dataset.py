import os
import config
import numpy
import TokenizedLyricsDataset

def create_datasets():
  root = config.DB_PATH
  paths = []
  for folder_name in os.listdir(root):
    folder = os.path.join(root, folder_name)
    if os.path.isdir(folder):
      for filename in os.listdir(folder):
          file_path = os.path.join(folder, filename)
          if os.path.isfile(file_path):
            paths.append(file_path)

  numpy.random.shuffle(paths)
  split_ind = int(len(paths) * config.TRAIN_VAL_SPLIT)
  train_paths = paths[:split_ind]
  val_paths = paths[split_ind:]

  train_dataset = TokenizedLyricsDataset(train_paths, True)
  val_dataset = TokenizedLyricsDataset(val_paths, False)

  return train_dataset, val_dataset