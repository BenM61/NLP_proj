import requests
import config
import os
import pickle

def template():   
	link = "https://www.azlyrics.com/lyrics/gabriellacilmi/sweetaboutme.html"
	response = requests.get(link)
	txt = response.text
	txt = txt.split("Sorry about that. -->")[1]
	txt = txt.split("<br><br>")[0]
	txt = txt.replace("<br>","").replace("</div>","").replace("<div>","")
	print(txt)
	response.close()

def change_file(genre_dict, songs_dict):
	## making the content we want to write
	contents = ["0" for i in range (0, 10)]
	contents[0] = "SUM OF ALL SONGS IN DATABASE: "+ genre_dict["all"] +"\n"
	contents[1] ="\n"
	contents[2] = "Pop songs: " + genre_dict["Pop"] +"\n"
	contents[3] = "Jazz songs: " + genre_dict["Jazz"] +"\n"
	contents[4] = "Hiphop songs: " + genre_dict["Hiphop"] +"\n"
	contents[5] = "Rock songs: " + genre_dict["Rock"] +"\n"
	contents[6] = "Classical songs: " + genre_dict["Classical"] +"\n"
	contents[7] = "R&b songs: " + genre_dict["R&b"] +"\n"
	contents[8] = "Blues songs: " + genre_dict["Blues"] +"\n"
	contents[9] = "Electronic songs: " + genre_dict["Electronic"] +"\n"

	## actually writing to the file 
	f = open("create_db/genres.txt", "w")
	new_file_contents = "".join(contents)
	f.write(new_file_contents)

# title- song title / genre list
def load_dict():
	# print("[DEBUG] loading temp dict")


	title_dict = {}


	if os.path.exists(config.TEMP_TITLES_GENRES_PATH):
		title_file = open(config.TEMP_TITLES_GENRES_PATH, "rb")

		title_dict = pickle.load(title_file)

	return title_dict

# title- song title / genre list
def save_dict(title_dict):
	# print("[DEBUG] saving temp dicts")

	title_file = open(config.TEMP_TITLES_GENRES_PATH, "wb")

	pickle.dump(title_dict, title_file)

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

# used to find songs with unwanted names/ contents/ etc.
def get_invalid(label, criterion):
	folder_path = config.LABEL_TO_PATH[label]
	bad_files_names = []
	for filename in os.listdir(folder_path):
		file_path = os.path.join(folder_path, filename)
		if os.path.isfile(file_path):
			f = open(str(file_path), "r")
			if not criterion(f):
				bad_files_names.append(filename)
				
	return bad_files_names

def remove_invalid_songs(label):
	title_dict = load_dict()
	label_folder = config.LABEL_TO_PATH[label]

	invalid_fnames = get_invalid(label, is_song_file_valid)

	print(f"[INFO] Removing {len(invalid_fnames)} invalid {label} songs")

	for fname in invalid_fnames:
		curr_title = fname[:-4]
		curr_path = os.path.join(label_folder, fname)

		os.remove(curr_path)
		if (title_dict[curr_title] == [label]):
			del title_dict[curr_title]
		else:
			title_dict[curr_title].remove(label)
		
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