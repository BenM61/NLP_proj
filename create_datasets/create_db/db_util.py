import requests
import config
import os
import pickle

def get_status():
	labels = config.LABELS

	label_dict = {}
	cnt = 0
	for label in labels:
		folder_path = config.LABEL_TO_PATH[label]
		curr_cnt = 0
		if os.path.exists(folder_path):
			for filename in os.listdir(folder_path):
				file_path = os.path.join(folder_path, filename)
				if os.path.isfile(file_path):
					curr_cnt += 1

		label_dict[label] = curr_cnt
		cnt += curr_cnt
	
	label_dict["all"] = cnt
	total_titles = len(load_dict().keys())
	## making the content we want to write
	s = "STATUS:\n"
	s += "AMONUT OF ALL SONGS IN DATABASE (including duplicates): "+ str(label_dict["all"]) +"\n"
	s += "AMONUT OF ALL SONGS IN DATABASE (excluding duplicates): "+ str(total_titles) +"\n\n"
	for label in config.LABEL_TO_PATH.keys():
		amt = label_dict[label]
		percent = str(label_dict[label] * 100 / total_titles)[:4]
		s += f"{label} songs: {percent}% ({amt})\n"

	genre_amt_dict = {}
	for v in load_dict().values():
		n = len(v)
		if n not in genre_amt_dict:
			genre_amt_dict[n] = 1
		else:
			genre_amt_dict[n] += 1

	s += f"\n\nTag distribution: \n"
	for i in range(len(genre_amt_dict.keys())):
			amt = genre_amt_dict[i+1]
			percent = str(genre_amt_dict[i+1]*100 / total_titles)[:4]
			s += f"{i+1}: {percent}% ({amt})   "

	s += "\n"

	return s

# title- song title / genre list
def load_dict():
	# print("[DEBUG] loading temp dict")

	title_dict = {}

	if os.path.exists(config.TITLES_GENRES_PATH):
		title_file = open(config.TITLES_GENRES_PATH, "rb")

		title_dict = pickle.load(title_file)
		title_file.close()
		return title_dict

	if os.path.exists(config.TEMP_TITLES_GENRES_PATH):
		title_file = open(config.TEMP_TITLES_GENRES_PATH, "rb")

		title_dict = pickle.load(title_file)
		title_file.close()
	return title_dict

# title dict- song title / genre list
def save_dict(dict, path=config.TEMP_TITLES_GENRES_PATH):
	# print("[DEBUG] saving temp dicts")

	title_file = open(path, "wb")

	pickle.dump(dict, title_file)
	title_file.close()

def save_stats():
	with open(config.OVERALL_STATS_PATH, "w") as f:
		f.write(get_status())

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
def get_invalid(criterion, label):
	folder_path = config.LABEL_TO_PATH[label]
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
def remove_invalid(criterion, label="all"):
	if label=="all":
		labels = config.LABELS
	else:
		labels = [label]

	invalid_fpaths = []
	title_dict = load_dict()
	for curr_label in labels:
		curr_invalid = get_invalid(criterion, curr_label)
		invalid_fpaths += curr_invalid
		print(f"[INFO] Removing {len(curr_invalid)} invalid {curr_label} songs")

	for fpath in invalid_fpaths:
		curr_title = fpath.split("/")[-1][:-4]
		curr_label = fpath.split("/")[-2]
		os.remove(fpath)
		if (title_dict[curr_title] == [curr_label]):
			del title_dict[curr_title]
		else:
			title_dict[curr_title].remove(curr_label)
		
	save_dict(title_dict)

# get titles of songs that doesn't have lyrics files from the dictionaries
# used for fixing incorrect counting
def get_ghost_songs(label):
	title_dict = load_dict()
	folder_path = config.LABEL_TO_PATH[label]
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
		title_dict = load_dict()

	labels = title_dict[title]
	contents = {}
	for label in labels:
		dir_path = config.LABEL_TO_PATH[label]
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
					dir_path = config.LABEL_TO_PATH[in_label]
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