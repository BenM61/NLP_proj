import os
from torch.utils.data import Dataset

class LyricsDataset(Dataset):
	def _get_title(self, song_path):
		return str(song_path).split(os.path.sep)[-1][:-4]

	def _get_lyrics(self, song_path):
		with open(song_path, "r") as f:
			lyrics = f.read()
	
		return lyrics

	def _get_labels(self, title_dict, song_path):
		title = str(song_path).split(os.path.sep)[-1][:-4]
		unpadded_labels = title_dict[title]

		labels = unpadded_labels + [""] * (5 - len(unpadded_labels))

		return labels

	def __init__(self, title_dict, songs_paths, ignore_titles):
		super(LyricsDataset, self).__init__()
		
		self.ignore_titles = ignore_titles
		self.lyrics = []
		self.labels = []
		self.titles = []

		for path in songs_paths:
			lyrics = self._get_lyrics(path)
			label = self._get_labels(title_dict, path)

			title = self._get_title(path)

			self.lyrics.append(lyrics)
			self.labels.append(label)
			self.titles.append(title)
					
		self.num_songs = len(self.labels)

	def __len__(self):
		return self.num_songs
	
	def __getitem__(self, index):
		title = self.titles[index] if self.ignore_titles else ""
		return {"lyrics" : self.lyrics[index] + title,
						"label" : self.labels[index][0]}
