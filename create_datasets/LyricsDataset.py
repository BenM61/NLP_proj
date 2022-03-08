import os
from create_datasets import config_dataset as config
from torch.utils.data import Dataset


class LyricsDataset(Dataset):
	def _get_title(self, song_path):
		return str(song_path).split(os.path.sep)[-1][:-4].replace("_", " ")

	def _get_lyrics(self, song_path):
		with open(song_path, "r") as f:
			return f.read()

	def _get_labels(self, title_dict, song_path):
		title = str(song_path).split(os.path.sep)[-1][:-4]
		unpadded_labels = title_dict[title]
		return unpadded_labels + [""] * (5 - len(unpadded_labels))

	def __init__(self, title_dict, songs_paths, ignore_titles):
		super(LyricsDataset, self).__init__()

		self.num_songs = len(songs_paths)
		self.ignore_titles = ignore_titles
		self.titles = []
		self.lyrics = []
		self.labels = []
		for path in songs_paths:
			title = self._get_title(path)
			lyrics = self._get_lyrics(path)
			label = self._get_labels(title_dict, path)

			self.titles.append(title)
			self.lyrics.append(lyrics)
			self.labels.append(label)

	def __len__(self):
		return self.num_songs
	
	def __getitem__(self, index):
		if self.ignore_titles:
			return self.lyrics[index], self.labels[index]
		else:
			return (self.titles[index], self.lyrics[index]), self.labels[index]