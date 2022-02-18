import requests 
import db_util
import urllib3

urllib3.disable_warnings()

genre_dict ={}
song_dict = {}
genre_dict, song_dict = db_util.get_dict()

def getFile():
    main_link = "https://www.lyrics.com/genre/Rock"
    j = 0
    for i in range (1,21):
        if i > 1:
            link = main_link +"/"+ str(i)
        content = requests.get(main_link, verify=False)
        content = content.text
        content = content.split("<p class=\"lyric-meta-title\">")[1:]
        for c in content:
            c = c.split("href=")[1]
            c = c.split(">")[0]
            c = c.replace("\"","")
            print(c)
            getLyrics(c)
        return
        
def getLyrics(link):
    link = "https://www.lyrics.com/" + link
    response = requests.get(link, verify=False)
    txt = response.text

    name = txt.split(" | ")[0]
    name = name.split("<title>")[1]
    singer = name.split(" - ")[0].strip()
    name = name.split(" - ")[1].split("Lyrics")[0].strip() +"("+singer+")"

    if name in song_dict:
        if "rock" in song_dict[name]: # exist in db
            return
        else: #exist on another genre
            song_dict[name] = song_dict[name] + ["rock"]
    else:
        song_dict[name] = ["rock"]
        genre_dict["all"] = str (int(genre_dict["all"]) + 1)
        
    genre_dict["rock"] = str (int(genre_dict["rock"]) + 1)

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
    with open("create_db/rock.txt", "a") as f:
        f.write("name: "+ name +":\n")
        f.write(clean_ly+"\n\n")

getFile()
print("done")
print(genre_dict)
