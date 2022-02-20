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

	#contents = ["0" for i in range (0,len(songs_dict))]
	#i = 0
	#for key in songs_dict.keys():
	#    contents[i] = key +": " + ", ".join(songs_dict[key]) +"\n"
	#    i += 1
	
	#f = open("create_db/titles.txt", "w")
	#new_file_contents = "".join(contents)
	#f.write(new_file_contents[:-1])

def load_dicts():
	# print("[DEBUG] loading temp dicts")

	stats_dict = {"all" : 0, "POP" : 0, "HIP_HOP" : 0, "ROCK" : 0,
			"ELECTRONIC" : 0, "BLUES" : 0, "JAZZ" : 0, "CLASSICAL" : 0}
	#  TODO:  and so on for the other categories...

	title_dict = {}


	if os.path.exists(config.TEMP_OVERALL_STATS_PATH):
		stats_file = open(config.TEMP_OVERALL_STATS_PATH, "rb")
		title_file = open(config.TEMP_TITLES_GENRES_PATH, "rb")

		stats_dict = pickle.load(stats_file)
		title_dict = pickle.load(title_file)

	return stats_dict, title_dict

# state- overall count, genre/ int
# title- song title / 
def save_dicts(stats_dict, title_dict):
	# print("[DEBUG] saving temp dicts")

	stats_file = open(config.TEMP_OVERALL_STATS_PATH, "wb")
	title_file = open(config.TEMP_TITLES_GENRES_PATH, "wb")

	pickle.dump(stats_dict, stats_file)
	pickle.dump(title_dict, title_file)

def is_char_valid(c):
	c = str(c)
	return c == " " or c == "-" or c == "_" or (c.isascii() and c.isalnum())

def get_valid_filename(name):
	name = str(name)
	name2 = "".join(filter(is_char_valid, name))
	
	if not name[0].isascii() or not name == name2:
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
def get_invalid(folder_path, criterion):
	cnt, cnt_bad = 0, 0
	bad_files_names = []
	for filename in os.listdir(folder_path):
		file_path = os.path.join(folder_path, filename)
		if os.path.isfile(file_path):
			cnt += 1
			f = open(str(file_path), "r")
			if not criterion(f):
				cnt_bad += 1
				bad_files_names.append(filename)
				
	print(f"[DEBUG] There are {cnt_bad} invalid songs out of {cnt} total")
	return bad_files_names

def remove_invalid_songs(path, criterion):
	invalid_names = get_invalid(path, criterion)
	amount = len(invalid_names)

	print(f"[INFO] Deleting {amount} invalid songs")

	for name in invalid_names:
		file_path = os.path.join(path, name)
		os.remove(file_path)

	print(f"[INFO] Delete complete")