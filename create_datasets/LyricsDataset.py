from create_datasets import config_dataset as config
from .create_db.file_utils import save_dict, save_datasets_stats
import os
from torch.utils.data import Dataset
from transformers import T5Tokenizer
import numpy as np

class LyricsDataset(Dataset):
	def _get_title(self, song_path):
		return str(song_path).split(os.path.sep)[-1][:-4]

	def _get_lyrics(self, song_path):
		with open(song_path, "r") as f:
			raw_lyrics = f.read()
		
		if not self.ignore_titles:
			raw_lyrics = self._get_title(song_path).replace("_", " ") + "\n\n\n" + raw_lyrics
		
		lyrics = self._tokenizer.encode_plus(
															raw_lyrics, 
															max_length=config.TOKENIZER_SOURCE_MAX_LENGTH,
															padding="max_length",
															truncation=False,
															return_attention_mask=True,
															return_token_type_ids=False,
															return_tensors='pt'
														)

		return raw_lyrics, lyrics

	def _get_labels(self, title_dict, song_path):
		title = str(song_path).split(os.path.sep)[-1][:-4]
		unpadded_labels = title_dict[title]

		raw_labels = unpadded_labels + [""] * (5 - len(unpadded_labels))

		raw_labels_string= " , ".join(raw_labels)
		labels = self._tokenizer.encode_plus(
															raw_labels_string, 
															max_length=config.TOKENIZER_TARGET_MAX_LENGTH,
															padding="max_length",
															truncation=False,
															return_attention_mask=True,
															return_token_type_ids=False,
															return_tensors='pt'
														)
		return raw_labels, labels

	def __init__(self, title_dict, songs_paths, ignore_titles):
		super(LyricsDataset, self).__init__()
		
		self._tokenizer = config.TOKENIZER
		self.ignore_titles = ignore_titles
		self.raw_lyrics = []
		self.raw_labels = []
		self.lyrics = []
		self.labels = []
		
		for path in songs_paths:
			raw_lyrics, lyrics = self._get_lyrics(path)
			raw_label, label = self._get_labels(title_dict, path)

			raw_title = self._get_title(path)

			# remove songs with very long tokenization
			if np.shape(lyrics['input_ids'])[1] > config.TOKENIZER_SOURCE_MAX_LENGTH:
				del title_dict[raw_title]
			else:
				self.raw_lyrics.append(raw_lyrics)
				self.raw_labels.append(raw_label)
				self.lyrics.append(lyrics)
				self.labels.append(label)

		self.num_songs = len(self.labels)
		save_dict(title_dict, config.TITLES_GENRES_PATH)
		save_datasets_stats(config.OVERALL_STATS_PATH, title_dict)

	def __len__(self):
		return self.num_songs
	
	def __getitem__(self, index):
		return {"lyrics" : self.lyrics[index], 
						"labels" : self.labels[index]}
