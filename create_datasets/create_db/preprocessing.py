import os

def preprocess(lyrics):
    chorDec = True
    isChorus = False
    isRep = False
    chor =[]
    repBlock = []
    new_song = []
    repNum = 0
    lines = lyrics.split("\n")
    lines = [l + "\n" for l in lines]
    #if lines[-1] != "\n":
    #    lines += "\n"

    for line in lines:
        line = line.replace("[","(").replace("]",")")
        line = line.replace("..","...")
        while True:
            if ("...." in line):
                line = line.replace("....","...")
            else:
                break

        l = line.replace("\n","").replace("(","").strip().upper()

        #[chorus:]
        if l.startswith("CHORUS") or l.startswith("REPEAT CHORUS"):
            if chorDec: #first time we seeing the chorus
                isChorus = True
                chorDec = False
            else: #repeat chorus
                repeat = 1
                if "X" in l or "REPEAT" in l:
                    for letter in l:
                        if letter.isdigit():
                            repeat = max(int(letter)-1,1)
                for j in range(0,repeat):
                    for c in chor:
                        new_song.append(c)
                    if j != repeat-1:
                        new_song.append("\n")
        else:
            repeat = 1

            #[repeat: x3]
            if "REPEAT" in l:
                for letter in l:
                    if letter.isdigit():
                        repeat = max(int(letter)-1,1)

                line = line.split("[")[0].strip()
                line = line.split("(Repeat")[0].strip()

                #bla bla bla [repeat: x3]
                if line != "":
                    line = line + "\n"
                    for j in range(0,repeat):
                        new_song.append(line)

                # [repeat: x4]
                # bla 
                # bla

                else:
                    isRep = True
                    repNum = repeat

            #regular lines
            new_song.append(line)
            if (isChorus):
                if (line == "\n" or line == "\r\n"):
                    isChorus = False
                else:
                    chor.append(line)
            
            if(isRep):
                if (line == "\n" or line == "\r\n") and len(repBlock)>0:                     
                    isRep = False
                    repBlock.append(line)
                    for j in range (0,repNum):
                        for b in repBlock:
                            new_song.append(b)
                    repBlock = []
                else:
                    repBlock.append(line)

    if (new_song[-1] == "\n"):
        new_song = new_song[:-1]
    
    if (is_valid_song(new_song) == False):
        #print(new_song)
        return new_song, False
    return new_song, True

#check there is no text like [chorus] or [repeat] and this song is only lyrics
# AFTER we have done the preprocess
def is_valid_song(lyrics):
    if (len(lyrics) == 0): #empty song
        return False
    verses = 0
    lines_in_verse = 0
    words = 0
    for line in lyrics:
        words += len(line.split())
        line = line.upper()
        if "CHORUS" in line or "REPEAT" in line:
            return False
        for i in range(1,10):
            if ("X"+str(i)) in line:
                return False
            if ("VERSE "+str(i)) in line:
                return False
        if (line == "\n" or line == "\r\n"):
            if (lines_in_verse > 2):
                verses += 1
            lines_in_verse = 0
        else:
            lines_in_verse += 1
            
    if verses > 1 and (words >= 80 and words <= 400):
        return True

    return False
