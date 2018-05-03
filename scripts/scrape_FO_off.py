import requests

pages = ["ncaaoff2014", "ncaaoff2015", "ncaaoff2016", "ncaaoff"]
header = "Year,Team,Off S&P+,Off rank,Rush S&P+,Rush rank,Pass S&P+,Pass rank,SD S&P+,SD rank,PD S&P+,PD rank,Success rt+,Success rank,IsoPPP+,IsoPPP+ rank"

with open("s_and_p_off.csv", "w") as f:
    f.write(header+"\n")

    year = 2014
    for page in pages:
        r = requests.get("http://www.footballoutsiders.com/stats/" + page)
        data = r.content
        table = data[data.find("<table"):data.find("</table")].split("\n")

        col_count = 0
        for line in table:
            if line.find("<td>") != -1 or line.find("<td align=\"r") != -1:
                col_data = line[line.find(">")+1:line.find("</td")].strip()
                if col_count == 0:
                    f.write(str(year)+","+col_data+",")
                    col_count += 1
                elif col_count < 14:
                    if col_data[0] == '<':
                        col_data = col_data[col_data.find(">")+1:col_data.find("</b")].strip()
                    if page == "ncaaoff" and col_count == 3:
                        success_rt = col_data
                    elif page == "ncaaoff" and col_count == 4:
                        success_rank = col_data
                    elif page == "ncaaoff" and col_count == 5:
                        isoPPP = col_data
                    elif page == "ncaaoff" and col_count == 6:
                        isoRank = col_data
                    else:
                        f.write(col_data+",")
                    col_count += 1
                else:
                    if page == "ncaaoff":
                        f.write(col_data+","+success_rt+","+success_rank+","+isoPPP+","+isoRank+"\n")
                    else:
                        f.write(col_data+"\n")
                    col_count = 0
        year += 1
