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
                genre_dict["Pop"] = line[11:-1]
                continue
            if i == 3: #Jazz
                genre_dict["Jazz"] = line[12:-1]
                continue
            if i == 4: #hiphop
                genre_dict["Hiphop"] = line[14:-1]
                continue
            if i == 5: #rock
                genre_dict["Rock"] = line[12:-1]
                continue
            if i == 6: #Classical
                genre_dict["Classical"] = line[17:-1]
                continue
            if i == 7: #r$b
                genre_dict["R&b"] = line[11:-1]
                continue
            if i == 8: #Blues
                genre_dict["Blues"] = line[13:-1]
                continue
            if i == 9: #electronic
                genre_dict["Electronic"] = line[18:-1]

    with open("create_db/titles.txt", "r") as f:
        list = f.readlines()
        for line in list:         
            list2 = line.split(": ")
            if len(list2) == 0: #last line of the file
                break
            song = list2[0]
            genre = list2[1].replace("\n","").strip()
            if song not in songs_dict:
                songs_dict[song] = [genre]
            else:
                songs_dict[song] += [genre]
            i+=1

    return genre_dict, songs_dict

#dict1, dict2 = get_dict()
#change_file(dict1, dict2)
