import requests
from transformers import get_polynomial_decay_schedule_with_warmup
import config_db
import os
import pickle

def get_label_counts(root):
	#how much songs from each label
	label_dict = {}
	cnt = 0
	for folder_name in os.listdir(root):
		folder_path = os.path.join(root, folder_name)
		if os.path.isdir(folder_path):
			curr_cnt = 0
			for filename in os.listdir(folder_path):
				file_path = os.path.join(folder_path, filename)
				if os.path.isfile(file_path):
					curr_cnt += 1

			label_dict[folder_name] = curr_cnt
			cnt += curr_cnt
	
	return cnt, label_dict

def create_title_dict(dict_path=config_db.TITLES_GENRES_PATH):
	titles_dict = {}
	for genre in os.listdir(config_db.DB_PATH):
		genre_folder = os.path.join(config_db.DB_PATH, genre)
		if os.path.isdir(genre_folder):
			for song in os.listdir(genre_folder):
				song_txt = os.path.join(genre_folder, song)
				if os.path.isfile(song_txt):
					song = song[:-4]
					if (song in titles_dict):
						titles_dict[song].append(genre)
					else:
						titles_dict[song] = [genre]

	save_dict(titles_dict,dict_path)

def get_tag_distribution(dict_path):
	total_titles = len(load_dict(dict_path).keys())
	total_titles = len(load_dict(dict_path).keys())
	genre_amt_dict = {}
	for v in load_dict(dict_path).values():
		n = len(v)
		if n not in genre_amt_dict:
			genre_amt_dict[n] = 1
		else:
			genre_amt_dict[n] += 1
	return total_titles, genre_amt_dict

def get_status_message(cnt, total_titles, label_dict, genre_amt_dict=None):
	## making the content we want to write
	s = "STATUS:\n"
	s += "AMONUT OF ALL SONGS IN DATABASE (including duplicates): "+ str(cnt) +"\n"
	s += "AMONUT OF ALL SONGS IN DATABASE (excluding duplicates): "+ str(total_titles) +"\n\n"
	for label in label_dict.keys():
		amt = label_dict[label]
		percent = str(label_dict[label] * 100 / total_titles)[:4]
		s += f"{label} songs: {percent}% ({amt})\n"

	if not genre_amt_dict is None:
		s += f"\n\nTag distribution: \n"
		for i in range(len(genre_amt_dict.keys())):
				amt = genre_amt_dict[i+1]
				percent = str(genre_amt_dict[i+1]*100 / total_titles)[:4]
				s += f"{i+1}: {percent}% ({amt})   "

	s += "\n"

	return s

def get_status(root=config_db.DB_PATH, dict_path=config_db.TITLES_GENRES_PATH):

	cnt, label_dict = get_label_counts(root)
	total_titles, genre_amt_dict = get_tag_distribution(dict_path)
	s = get_status_message(cnt, total_titles, label_dict, genre_amt_dict)

	return s

# title- song title / genre list
def load_dict(path=config_db.TEMP_TITLES_GENRES_PATH):
	# print("[DEBUG] loading temp dict")

	title_dict = {}

	if os.path.exists(path):
		title_file = open(path, "rb")

		title_dict = pickle.load(title_file)
		title_file.close()
	else:
		if os.path.exists(config_db.TITLES_GENRES_PATH):
			title_file = open(config_db.TITLES_GENRES_PATH, "rb")

			title_dict = pickle.load(title_file)
			title_file.close()

	return title_dict

# title dict- song title / genre list
def save_dict(dict, path=config_db.TEMP_TITLES_GENRES_PATH):
	# print("[DEBUG] saving temp dicts")

	title_file = open(path, "wb")

	pickle.dump(dict, title_file)
	title_file.close()

def save_stats(root_folder=config_db.DB_PATH, dict_path=config_db.TITLES_GENRES_PATH, to_path=config_db.OVERALL_STATS_PATH):
	with open(to_path, "w") as f:
		f.write(get_status(root_folder, dict_path=dict_path))

def is_char_valid(c):
	c = str(c)
	return c == " " or c == "-" or c == "_" or (c.isascii() and c.isalnum())

def get_valid_filename(name):
	name = str(name)
	name2 = "".join(filter(is_char_valid, name))
	
	if name == "" or not name[0].isascii() or not name == name2:
		return None

	name = name.replace(" ", "_")
	return name

def is_content_valid(lyrics):
	lyrics = str(lyrics)
	lyrics = lyrics.replace(":", "")
	lyrics = lyrics.replace("?", "")
	lyrics = lyrics.replace("!", "")
	lyrics = lyrics.replace("\"", "")
	lyrics = lyrics.replace(",", "")
	lyrics = lyrics.replace("-", "")
	lyrics = lyrics.replace("'", "")
	lyrics = lyrics.replace("(", "")
	lyrics = lyrics.replace(")", "")
	lyrics = lyrics.replace(".", "")
	lyrics = lyrics.replace("[", "")
	lyrics = lyrics.replace("]", "")
	lyrics = lyrics.replace(";", "")
	lyrics = lyrics.replace("&", "")
	lyrics = lyrics.replace("*", "")
	lyrics = "".join(lyrics.split())

	return lyrics.isascii() and lyrics.isalnum()

def is_song_file_valid(f):
	# filename without path and without .txt
	name = os.path.basename(f.name)[:-4]
	content = f.read()
	return is_content_valid(content) and (get_valid_filename(name) == name)

# used to find paths of songs with unwanted names/ contents/ etc.
def get_invalid(criterion, label, root):
	folder_path = os.path.join(root, label)
	bad_files_paths = []
	for filename in os.listdir(folder_path):
		file_path = os.path.join(folder_path, filename)
		if os.path.isfile(file_path):
			f = open(str(file_path), "r")
			if not criterion(f):
				bad_files_paths.append(file_path)
			f.close()
	return bad_files_paths

# used to delete paths of songs with unwanted names/ contents/ etc from label folder.
def remove_invalid(criterion, label="all", root=config_db.DB_PATH, 
										dict_file=config_db.TITLES_GENRES_PATH):
	if label=="all":
		labels = config_db.LABELS
	else:
		labels = [label]

	invalid_fpaths = []
	title_dict = load_dict(dict_file)
	for curr_label in labels:
		curr_invalid = get_invalid(criterion, curr_label, root)
		invalid_fpaths += curr_invalid

	for fpath in invalid_fpaths:
		curr_title = fpath.split(os.path.sep)[-1][:-4]
		curr_label = fpath.split(os.path.sep)[-2]
		os.remove(fpath)
		if (title_dict[curr_title] == [curr_label]):
			del title_dict[curr_title]
		else:
			title_dict[curr_title].remove(curr_label)
		
	save_dict(title_dict, dict_file)

# get titles of songs that doesn't have lyrics files from the dictionaries
# used for fixing incorrect counting
def get_ghost_songs(label):
	title_dict = load_dict()
	folder_path = config_db.LABEL_TO_PATH[label]
	bad_titles_names = []

	for title, genres in title_dict.items():
		if label in genres:
			filename = title + ".txt"
			file_path = os.path.join(folder_path, filename)
			if not os.path.isfile(file_path):
				bad_titles_names.append(title)
				print(f"[DEBUG] Found ghost- {title} in {label}")

	return bad_titles_names

def remove_ghost_songs(label):
	title_dict = load_dict()
	ghost_titles = get_ghost_songs(label)

	print(f"[INFO] Removing {len(ghost_titles)} ghost {label} songs from dicts")

	for title in ghost_titles:
		if (title_dict[title] == [label]):
			del title_dict[title]
		else:
			title_dict[title].remove(label)
	
	save_dict(title_dict)

# returns dict_values of a dict from content (of that title)
# to list of genres it's in
def get_version_distribution(title, title_dict=None):
	if title_dict is None:
		title_dict = load_dict(config_db.TITLES_GENRES_PATH)

	labels = title_dict[title]
	contents = {}
	for label in labels:
		dir_path = config_db.LABEL_TO_PATH[label]
		fpath = os.path.join(dir_path, title + ".txt")
		f = open(fpath, "r")
		content = f.read()
		if content in contents:
			contents[content].append(label)
		else:
			contents[content] = [label]
		f.close()
	return(contents.values())

# get paths to all files that has duplicates
def get_duplicates(title_dict):
	bad_fpaths = []
	
	for title in title_dict.keys():
		version_distribution = get_version_distribution(title, title_dict)
		if len(version_distribution) == 1:
			continue

		# reverse order to keep versions from the small categories
		version_distribution_rev = reversed(version_distribution)
		# keep the version with most apps
		keep_version = max(version_distribution_rev, key=len)
		# add all other versions to the bad path list
		for version in version_distribution:
			if not version == keep_version:
				for in_label in version:
					dir_path = config_db.LABEL_TO_PATH[in_label]
					fpath = os.path.join(dir_path, title + ".txt")
					bad_fpaths.append(fpath)

	return bad_fpaths

def remove_duplicates():
	title_dict = load_dict()
	invalid_fpaths = get_duplicates(title_dict)

	print(f"[INFO] removing {len(invalid_fpaths)} duplicate files")
	for fpath in invalid_fpaths:
		curr_title = fpath.split("/")[-1][:-4]
		curr_label = fpath.split("/")[-2]
		os.remove(fpath)
		if (title_dict[curr_title] == [curr_label]):
			del title_dict[curr_title]
		else:
			title_dict[curr_title].remove(curr_label)
		
	save_dict(title_dict)

def get_datasets_status(datasets_title_dict):
	cnt = len(datasets_title_dict.keys())
	
	label_dict = {}
	# count song amount per label
	for title in datasets_title_dict.keys():
		for label in datasets_title_dict[title]:
			if label in label_dict:
				label_dict[label] += 1
			else:
				label_dict[label] = 1

	# calculate distribution of label amount
	distribution = {}
	for title in datasets_title_dict.keys():
		n = len(datasets_title_dict[title])
		if n in distribution:
			distribution[n] += 1
		else: 
			distribution[n] = 1

	# set the status message
	s = "STATUS:\n" + f"AMONUT OF ALL SONGS IN DATABASE: {cnt}\n\n"
	for label in label_dict.keys():
		amt = label_dict[label]
		percent = str(label_dict[label] * 100 / cnt)[:4]
		s += f"{label} songs: {percent}% ({amt})\n"
	s += "NOTE: the precentages go above 100 because a song can have more than one genre\n"

	s += f"\n\nTag distribution: \n"
	for i in range(len(distribution.keys())):
			amt = distribution[i+1]
			percent = str(distribution[i+1]*100 / cnt)[:4]
			s += f"{i+1}: {percent}% ({amt})   "

	s += "\n"
	return s

def save_datasets_stats(stats_file_path, datasets_title_dict):
	with open(stats_file_path, "w") as f:
		f.write(get_datasets_status(datasets_title_dict))