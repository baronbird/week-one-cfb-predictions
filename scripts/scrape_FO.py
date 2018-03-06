import requests

pages = ["ncaa2013", "ncaa2014", "ncaa2015", "ncaa2016", "ncaa"]
header = "Team,Record,2ndO wins (diff),S&P+ (pctl),S&P+ (margin),Overall rank,Off S&P+,Off rank,Def S&P+,Def rank"

for page in pages:
    r = requests.get("http://www.footballoutsiders.com/stats/" + page)
    data = r.content
    table = data[data.find("<table"):data.find("</table")].split("\n")

    with open(page+".csv", "w") as f:
        f.write(header+"\n")

        col_count = 0
        for line in table:
            if line[0:4] == "<td>" or line[0:5] == "<td a":
                col_data = line[line.find(">")+1:line.find("</td")].strip()
                if col_count < 9:
                    if col_data[0] != '<':
                        f.write(col_data + ",")
                    else:
                        f.write(col_data[col_data.find(">")+1:col_data.find("</b")].strip()+",")
                    col_count += 1
                elif page == "ncaa2013" and col_count < 13:
                    col_count += 1
                elif col_count < 11:
                    col_count += 1
                else:
                    f.write(col_data+"\n")
                    col_count = 0
