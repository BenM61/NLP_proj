import os

#with open("create_db/Lyrics.txt", "a") as f:
#        f.write(name)

for i in range(1,8):
    print(i)
    genre=""
    if (i==1):
        genre = "BLUES"
    if (i==2):
        genre = "ELECTRONIC"
    if (i==3):
        genre = "HIP_HOP"
    if (i==4):
        genre = "POP"
    if (i==5):
        genre = "ROCK"
    if (i==6):
        genre = "FUNK" 
    if (i==7):
        genre = "JAZZ"
    
    directory = "create_db/DB/"+genre
    for song in os.listdir(directory):
        #song.txt
        with open(directory+"/"+song, "r") as f:
           lines = f.readlines()
        chorDec = True
        isChorus = False
        isRep = False
        chor =[]
        repBlock = []
        new_song = []
        repNum = 0
        if lines[-1] != "\n":
            lines.append("\n")
        for line in lines:
            l = line.replace("\n","").replace("[","").replace("(","").strip().upper()

            #[chorus:]
            if l.startswith("CHORUS") or l.startswith("REPEAT CHORUS") or l.startswith("CHORUS:REPEAT"):
                if chorDec: #first time we seeing the chorus
                    isChorus = True
                    chorDec = False
                else: #repeat chorus
                    repeat = 1
                    if "X" in l or "REPEAT" in l:
                        for letter in l:
                            if letter.isdigit():
                                repeat = int(letter) 
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
                    if line == "\n":
                        isChorus = False
                    else:
                        chor.append(line)
                
                if(isRep):
                    if line == "\n" and len(repBlock)>0:                     
                        isRep = False
                        repBlock.append(line)
                        for j in range (0,repNum):
                            for b in repBlock:
                                new_song.append(b)
                        repBlock = []
                    else:
                        repBlock.append(line)

        with open(directory+"/"+song, "w") as f:
            for line in new_song:
                f.write(line)




   
