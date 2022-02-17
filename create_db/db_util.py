import requests
    
link = "https://www.azlyrics.com/lyrics/gabriellacilmi/sweetaboutme.html"
response = requests.get(link)
txt = response.text
txt = txt.split("Sorry about that. -->")[1]
txt = txt.split("<br><br>")[0]
txt = txt.replace("<br>","").replace("</div>","").replace("<div>","")
print(txt)
response.close()