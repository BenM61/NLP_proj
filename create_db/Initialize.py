import requests 
import db_util
import urllib3
import config
import os

def getFile(Lyrics_genre, song_dict):
	genre_label = config.GENRE_TO_LABEL[Lyrics_genre]
	genre_folder = config.LABEL_TO_PATH[genre_label]

	if not os.path.exists(genre_folder):
		os.makedirs(genre_folder)

	for i in range (1, 500):
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
			getLyrics(c, genre_label, genre_folder, song_dict) 

		# update dicts after each page
		db_util.save_dict(song_dict)
		
def getLyrics(link, genre, folder, song_dict):
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

	if name in song_dict:
		if genre in song_dict[name]: # exist in db
			return
		else: # exist on another genre
			song_dict[name] = song_dict[name] + [genre]
	else:
		song_dict[name] = [genre]

	song_path = os.path.join(folder, name + ".txt")
	f = open(song_path, "w")
	f.write(clean_ly)

def initialize():
	urllib3.disable_warnings()

	if not os.path.exists(config.DB_PATH):
		os.makedirs(config.DB_PATH)

	title_dict = db_util.load_dict()

	#getFile("Pop", title_dict) #1887
	#getFile("Hip%20Hop", title_dict) #596
	#getFile("Rock", title_dict) #2668
	getFile("Electronic", title_dict) #881
	#getFile("Blues", title_dict) #217
	#getFile("Classical", title_dict) #34
	#getFile("Jazz", title_dict) #668

	print("[INFO] Finished initialization")

def get_status():
	labels = ["POP", "HIP_HOP", "ROCK", "ELECTRONIC", "BLUES", "JAZZ", "CLASSICAL"]
	#  TODO:  and so on for the other categories...

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
	total_titles = len(db_util.load_dict().keys())
	## making the content we want to write
	s = "STATUS:\n"
	s += "SUM OF ALL SONGS IN DATABASE (including duplicates): "+ str(label_dict["all"]) +"\n"
	s += "SUM OF ALL SONGS IN DATABASE (excluding duplicates): "+ str(total_titles) +"\n\n"
	s += "Pop songs: " + str(label_dict["POP"]) +"\n"
	s += "Jazz songs: " + str(label_dict["JAZZ"]) +"\n"
	s += "Hiphop songs: " + str(label_dict["HIP_HOP"]) +"\n"
	s += "Rock songs: " + str(label_dict["ROCK"]) +"\n"
	s += "Classical songs: " + str(label_dict["CLASSICAL"]) +"\n"
	#s += "R&b songs: " + label_dict["R&b"] +"\n"
	s += "Blues songs: " + str(label_dict["BLUES"]) +"\n"
	s += "Electronic songs: " + str(label_dict["ELECTRONIC"]) +"\n"
	print(s)

	on = 0
	tw = 0
	tr = 0
	for k, v in db_util.load_dict().items():
		if len(v) == 1:
			on += 1
		elif len(v) ==2 :
			tw += 1
		elif len(v) == 3:
			tr += 1

	print(f"#One tag: {on}; #Two: {tw}; #Three: {tr}")


get_status()



#initialize()
