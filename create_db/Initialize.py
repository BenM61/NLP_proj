import requests 
import db_util
import urllib3
import config
import os

def getFile(Lyrics_genre, genre_dict, song_dict):
	genre_label = config.GENRE_TO_LABEL[Lyrics_genre]
	genre_folder = config.LABEL_TO_PATH[genre_label]

	if not os.path.exists(genre_folder):
		os.makedirs(genre_folder)

	for i in range (466, 500):
		main_link = "https://www.lyrics.com/genre/" + Lyrics_genre

		print(f"[INFO] Current page: {i}, current genre: {genre_label}")

		if i > 1:
			main_link += "/"+ str(i)
		if Lyrics_genre == "Pop" and i > 1887 :
			return
		if Lyrics_genre == "Hip%20Hop" and i > 596:
			return
		if Lyrics_genre == "Electronic" and i > 881:
			return
		if Lyrics_genre == "Blues" and i > 217:
			return
		if Lyrics_genre == "Classical" and i > 34:
			return
		if Lyrics_genre == "Jazz" and i > 668:
			return

		content = requests.get(main_link, verify=False)
		content = content.text
		content = content.split("<p class=\"lyric-meta-title\">")[1:]
		for c in content:
			c = c.split("href=")[1]
			c = c.split(">")[0]
			c = c.replace("\"","")
			getLyrics(c, genre_label, genre_folder, genre_dict, song_dict) 

		# update dicts after each page
		db_util.save_dicts(genre_dict, song_dict)
		
def getLyrics(link, genre, folder, genre_dict, song_dict):
	link = "https://www.lyrics.com/" + link
	response = requests.get(link, verify=False)
	txt = response.text

	name = txt.split(" | ")[0]
	name = name.split("<title>")[1]
	name = name.split(" - ")[1].split("Lyrics")[0].strip()

	# don't deal with titles that are invalid filenames
	orig_name = name
	name = db_util.get_valid_filename(name)
	if name is None:
		print(f"[DEBUG] skipped {orig_name}")
		return

	if name in song_dict:
		if genre in song_dict[name]: # exist in db
			return
		else: # exist on another genre
			song_dict[name] = song_dict[name] + [genre]
	else:
		song_dict[name] = [genre]
		genre_dict["all"] = str (int(genre_dict["all"]) + 1)
			
	genre_dict[genre] = str (int(genre_dict[genre]) + 1)
	ly = txt.split("data-lang=\"en\">")[1]
	ly = ly.split("</pre>")[0]
	ly = ly.split("</a>")
	clean_ly =""
	f = True
	for l in ly:
		for i in l:
			if i == "<":
				f = False
			if i == ">":
				f = True
				continue
			if f:
				clean_ly += i   

	response.close()

	# for valid names, write song lyrics if valid
	if not db_util.is_content_valid(clean_ly):
		print(f"[DEBUG] skipped {orig_name}, for content in non-English")
		return

	song_path = os.path.join(folder, name + ".txt")
	f = open(song_path, "w")
	f.write(clean_ly)

def initialize():
	urllib3.disable_warnings()

	if not os.path.exists(config.DB_PATH):
		os.makedirs(config.DB_PATH)

	stats_dict, title_dict = db_util.load_dicts()

	#getFile("Pop", stats_dict, title_dict) #1887
	getFile("Hip%20Hop", stats_dict, title_dict) #596
	#getFile("Rock", stats_dict, title_dict) #2668
	#getFile("Electronic", stats_dict, title_dict) #881
	#getFile("Blues", stats_dict, title_dict) #217
	#print(stats_dict)
	#stats_dict["CLASSICAL"] = 0
	#getFile("Classical", stats_dict, title_dict) #34
	#getFile("Jazz", stats_dict, title_dict) #668

	print("[INFO] Finished initialization")

def get_status():
	stats_dict, title_dict = db_util.load_dicts()
	print(stats_dict)

get_status()
initialize()
