import os
from torch.utils.data import Dataset

class LyricsDataset(Dataset):
	def _get_title(self, song_path):
		return str(song_path).split(os.path.sep)[-1][:-4]

	def _get_lyrics(self, song_path):
		with open(song_path, "r") as f:
			lyrics = f.read()
	
		return lyrics

	def _get_label(self, title_dict, song_path):
		title = str(song_path).split(os.path.sep)[-1][:-4]
		label = title_dict[title][0]

		return label

	def __init__(self, title_dict, songs_paths, ignore_titles, labels):
		super(LyricsDataset, self).__init__()
		self.num2label = {k: v for k, v in enumerate(labels)}
		self.label2num = {v: k for k, v in self.num2label.items()}
		
		self.ignore_titles = ignore_titles
		self.genre_songs_matrix = [[] for _ in self.num2label.keys()]

		for path in songs_paths:
			lyrics = self._get_lyrics(path)
			label = self._get_label(title_dict, path)
			title = self._get_title(path)

			if label in labels:
				index = self.label2num[label]
				self.genre_songs_matrix[index].append((lyrics, title))

		smallest_label = min([len(arr) for arr in self.genre_songs_matrix])
		self.num_in_label = smallest_label
		self.num_songs = smallest_label * len(self.label2num.keys())
		print(f"Songs in ds: {self.num_songs}; in each label: {self.num_in_label}")

	def __len__(self):
		return self.num_songs
	
	def __getitem__(self, index):
		label_index = index // self.num_in_label
		index = index % self.num_in_label
		title = self.genre_songs_matrix[label_index][index][1] if self.ignore_titles else ""

		return {"lyrics" : self.genre_songs_matrix[label_index][index][0] + title,
						"label" : label_index}
