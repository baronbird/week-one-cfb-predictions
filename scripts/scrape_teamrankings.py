# scrape_teamrankings.py
# scrape 144 team stats from teamrankings.com
# author: Sam Berning (@baronbird)

import requests
import sys
from HTMLParser import HTMLParser

master_stats = {}

def construct_url(href, year):
    return "https://www.teamrankings.com"+href+"?date="+str(year+1)+"-01-10"

class StatListParser(HTMLParser):
    inlist = False
    hrefready = False
    dataready = False
    stats = []
    hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "ul":
            if self.inlist:
                self.hrefready = True
            else:
                for attr in attrs:
                    if attr[0] == "class" and attr[1] == "chooser-list":
                        self.inlist = True
        if self.hrefready:
            if tag == "a":
                self.dataready = True
                for attr in attrs:
                    if attr[0] == "href":
                        self.hrefs.append(attr[1])
    def handle_endtag(self, tag):
        if tag == "ul":
            if self.hrefready:
                self.hrefready = False
            else:
                self.inlist = False
        if tag == "a":
            self.dataready = False
    def handle_data(self, data):
        if self.dataready:
            # had to hardcode for these features unfortunately
            if data == "Opponent Red Zone Scores per Game (TDs " or data == "Opponent Red Zone Scoring Percentage (TDs ":
                data += "and FGs)"
            self.stats.append(data)
            self.dataready = False

class CFBParser(HTMLParser):
    dataready = False
    year = 2013
    stat = "Yards Per Game"
    columns = 0
    team = {}

    def handle_starttag(self, tag, attrs):
        if tag == "td":
            self.columns += 1
            if self.columns <= 3:
                self.dataready = True
        if tag == "tr":
            self.columns = 0
    def handle_endtag(self, tag):
        if tag == "tr":
            if self.team == {}:
                return
            try:
                master_stats[self.year][self.team["name"]][self.stat] = self.team["stat"]
            except:
                master_stats[self.year][self.team["name"]] = {}
                master_stats[self.year][self.team["name"]][self.stat] = self.team["stat"]
            self.team = {}
        if tag == "td":
            self.dataready = False
    def handle_data(self, data):
        if self.dataready:
            if self.columns == 2:
                try:
                    self.team["name"] += data
                except:
                    self.team["name"] = data
            elif self.columns == 3:
                self.team["stat"] = data
    def handle_entityref(self, name):
        if name == "amp":
            self.team["name"] += "&"

if __name__ == "__main__":
    years = range(2013,2018)
    stat_links = {}

    for year in years:
        master_stats[year] = {}

    r = requests.get("https://www.teamrankings.com/college-football/stat/yards-per-game?date=2014-01-06")

    slp = StatListParser()
    slp.feed(r.content)

    for i in range(0, len(slp.stats)):
        stat_links[slp.stats[i]] = slp.hrefs[i]

    cfbp = CFBParser()

    stat_num = 1
    for stat, link in stat_links.items():
        for year in years:
            cfbp.year = year
            cfbp.stat = stat
            r = requests.get(construct_url(link, year))

            cfbp.feed(r.content)
        sys.stderr.write(str(stat_num)+". "+stat+"\n")
        stat_num += 1

    keys = list(master_stats[2013]["Notre Dame"].keys())
    print "Year, Team,", 
    for key in keys:
        print key+",",
    print
    for year, team in master_stats.items():
        for name, stats in team.items():
            print str(year)+", "+name+",",
            for key in keys:
                print stats[key]+",",
            print
