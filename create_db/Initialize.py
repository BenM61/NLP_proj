from operator import ge
import requests 
import db_util
import urllib3

urllib3.disable_warnings()

genre_dict ={}
song_dict = {}
genre_dict, song_dict = db_util.get_dict()

def getFile(genre):
    for i in range (1,2669):
        main_link = "https://www.lyrics.com/genre/" + genre
        print(i)
        if i > 1:
            main_link += "/"+ str(i)
        if genre == "Pop" and i > 1887 :
            return
        if genre == "Hip%20Hop" and i > 596:
            return
        if genre == "Electronic" and i > 881:
            return
        if genre == "Blues" and i > 217:
            return
        if genre == "Classical" and i > 34:
            return
        if genre == "Jazz" and i > 668:
            return
        content = requests.get(main_link, verify=False)
        content = content.text
        #print(main_link)
        content = content.split("<p class=\"lyric-meta-title\">")[1:]
        if len(content) == 0:
            print("check html on page " + str(i) + " of genre " + genre)
        for c in content:
            c = c.split("href=")[1]
            c = c.split(">")[0]
            c = c.replace("\"","")
            #print(c)
            getLyrics(c,genre)
        
def getLyrics(link,genre):
    if genre == "Hip%20Hop":
        genre = "Hiphop"
    link = "https://www.lyrics.com/" + link
    response = requests.get(link, verify=False)
    txt = response.text

    name = txt.split(" | ")[0]
    name = name.split("<title>")[1]
    singer = name.split(" - ")[0].strip()
    name = name.split(" - ")[1].split("Lyrics")[0].strip()

    if name in song_dict:
        if genre in song_dict[name]: # exist in db
            return
        else: #exist on another genre
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
    db_util.change_file(genre_dict, song_dict)
    with open("create_db/"+genre+".txt", "a") as f:
        name += "("+singer+")"
        f.write("name: "+ name +":\n")
        f.write(clean_ly+"\n\n")

getFile("Pop") #1887
getFile("Hip%20Hop") #596
getFile("Rock") #2668
getFile("Electronic") #881
getFile("Blues") #217
getFile("Classical") #34
getFile("Jazz") #668
print("done")
#print(genre_dict)
