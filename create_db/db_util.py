import requests

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
    contents = ["0" for i in range (0, len(songs_dict)+12)]
    contents[0] = "SUM OF ALL SONGS IN DATABASE: "+ genre_dict["all"] +"\n"
    contents[1] ="\n"
    contents[2] = "pop songs: " + genre_dict["pop"] +"\n"
    contents[3] = "rap songs: " + genre_dict["rap"] +"\n"
    contents[4] = "hiphop songs: " + genre_dict["hiphop"] +"\n"
    contents[5] = "rock songs: " + genre_dict["rock"] +"\n"
    contents[6] = "metal songs: " + genre_dict["metal"] +"\n"
    contents[7] = "r&b songs: " + genre_dict["r&b"] +"\n"
    contents[8] = "children songs: " + genre_dict["children"] +"\n"
    contents[9] = "electronic songs: " + genre_dict["electronic"] +"\n"
    contents[10] = "\n"
    contents[11] = "songs list with genres:" +"\n"

    i = 12
    for key in songs_dict.keys():
        contents[i] = key +": " + ", ".join(songs_dict[key]) +"\n"
        i += 1

    ## actually writing to the file 
    list = [content for content in contents]
    f = open("create_db/genres.txt", "w")
    new_file_contents = "".join(list)
    f.write(new_file_contents[:-1])

def get_dict():
    genre_dict = {}
    songs_dict = {}
    i = -1
    with open("create_db/genres.txt", "r") as f:
        list = f.readlines()
        for line in list:
            i+=1
            if i == 0:
                genre_dict["all"] = line[30:-1]
                continue
            if i == 1: #/n
                continue
            if i == 2: #pop
                genre_dict["pop"] = line[11:-1]
                continue
            if i == 3: #rap
                genre_dict["rap"] = line[11:-1]
                continue
            if i == 4: #hiphop
                genre_dict["hiphop"] = line[14:-1]
                continue
            if i == 5: #rock
                genre_dict["rock"] = line[12:-1]
                continue
            if i == 6: #metal
                genre_dict["metal"] = line[13:-1]
                continue
            if i == 7: #r$b
                genre_dict["r&b"] = line[11:-1]
                continue
            if i == 8: #children
                genre_dict["children"] = line[16:-1]
                continue
            if i == 9: #electronic
                genre_dict["electronic"] = line[18:-1]
                continue
            if i > 11: #songs
                list2 = line.split(":")
                song = list2[0]
                genres = list2[1][1:].replace("\n","").split(", ")
                songs_dict[song] = genres
                continue
    return genre_dict, songs_dict

#dict1, dict2 = get_dict()
#change_file(dict1, dict2)
