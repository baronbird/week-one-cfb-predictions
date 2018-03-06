import requests
import re

FIRST_YEAR = 2013
LAST_YEAR = 2017

years = [str(year) for year in range(FIRST_YEAR,LAST_YEAR+1)]
header = "year,away_team,home_team,neutral_site?,away_pts,home_pts"

with open("../data/schedule.csv", "w") as f:
    f.write(header+"\n")

    for year in years:
        r = requests.get("http://www.espn.com/college-football/schedule/_/year/" + year)
        data = r.text.split('\n')
        line = [l for l in data if "college-football/team/_/id" in l][0]

        games = line.split("data-is-neutral-site")[1:]

        for game in games:
            if 'Canceled' in game or 'Postponed' in game:
                continue

            away_team, home_team = re.findall("<span>(.*?)</span>",game)[0:2]
            away_abbr, home_abbr = re.findall('<abbr title=".*?">(.*?)</abbr>',game)
            game_text = re.match('.*data-home-text="(at|vs)".*',game).group(1)
            away_score = re.match(".*"+away_abbr+" (\d*).*",game).group(1)
            home_score = re.match(".*"+home_abbr+" (\d*).*",game).group(1)

            if game_text == 'vs':
                neutral_site = '1'
            else:
                neutral_site = '0'

            f.write(year+","+away_team+","+home_team+","+neutral_site+","+away_score+","+home_score+"\n")
        

