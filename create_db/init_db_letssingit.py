import requests 

def getFile():
    prefix = "https://www.letssingit.com/songs/popular/"
    j = 0
    for i in range (1,21):
        page = prefix + str(i)
        content = requests.get(page, verify=False)
        content = content.text
        content = content.split("layout_d")[1]
        content = content.split("<div class=\"flex margin_small\">")[0] #the line we want
        content = content.split("class=high_profile")[0:-1]
        for song in content:
            j += 1
            print(j)
            link = song.split("<a href=")
            link = link[-1].replace("\n","").replace("\r","").replace(" ","").replace("\"","")
            getLyrics(str(link))
        
def getLyrics(link):
    response = requests.get(link, verify=False)
    txt = response.text
    ly = txt.split("id=lyrics>")[1]
    ly = ly.split("<div id=adslot_69_1")[0]
    noLy = ly.split("Unfortunately we don&#39;t have the lyrics for the song")
    if (len(noLy) > 1):
        ly = "no lyrics found\n"
    else:
        h = "<div class=lyrics_part_name>outro</div>"
        i = "<div class=lyrics_part_name>intro</div>"
        j = "<div class=lyrics_part_name>Verse 1</div>"
        k = "<div class=lyrics_part_name>Verse 2</div>"
        l = "<div class=lyrics_part_name>Pre-Chorus</div>"
        m = "<div class=lyrics_part_name>Chorus</div>"
        n = "<div class=lyrics_part_name>"

        a = "<br>"
        b = "<div id=lai_desktop_inline></div><div id=lai_mobile_inline></div>"
        c = "</div>"
        d = "<div class=lyrics_part_text>"
        e = "<span class=lyrics_parentheses>"
        f = "</span>"
        g = "<div class=lyrics_part_name>"
        o = "<span class=lyrics_speech>"
        p = "<span class=lyrics_question>"

        ly = ly.replace(h,"").replace(i,"").replace(j,"").replace(k,"").replace(l,"").replace(m,"").replace(n,"")
        ly = ly.replace(a,"").replace(b,"").replace(c,"").replace(d,"").replace(e,"").replace(f,"").replace(g,"")
        ly = ly.replace(o,"").replace("&#39;","'").replace(p,"")
    
    ly = "lyrics: " + ly +"\n"
    name = txt.split("title")[1]
    name = name.split("- ")[1]
    name = name.split("Lyrics")[0]
    name = name.split("[")[0]
    name = name.replace("&#39;","'")
    name = "name: " + name + "\n"
    singer = txt.split("title")[1]
    singer = singer.split(" -")[0]
    singer = singer.replace(">","").replace(", ...","").replace(" ...","").replace("&#39;","'")
    if (singer == "Imagine ..."):
        singer = " Imagine Dragons"
    singer = "singer: " + singer + "\n"
    genre = txt.split("Genre:")[1]
    genre = genre.split("<div>")[1]
    genre = genre.split("</div>")[0]
    if (genre == "&minus;"):
        genre = "no genre found"
    genre = "genre: " + genre + "\n"
    
    response.close()
    with open("Lyrics.txt", "a") as f:
        f.write(name)
        f.write(singer)
        f.write(genre)
        f.write(ly)

getFile()
print("done")

#pop, rap, hiphop, rock, metal, r&b, children, electronic