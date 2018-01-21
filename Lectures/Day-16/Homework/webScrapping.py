@@ -1,41 +0,0 @@
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 11:13:56 2017

@author: baradhwaj
"""

import requests
from bs4 import BeautifulSoup
from lxml import html


req = requests.get("http://www.mationalytics.com/sample.html")
soup = BeautifulSoup(req.content)

for i in soup.find_all('li'):
    print(i.text)

    
with open(r'D:\mationalytics\WebScraping\sample.html', "r") as f:
page = f.read()


#############3 Get  Tables from page ##########

import pandas as pd
link = 'https://www.icc-cricket.com/rankings/mens/team-rankings/test'
html = requests.get(link).content
soup = BeautifulSoup(html)
temp_df = pd.read_html(str(soup))[0]

links = ["https://www.icc-cricket.com/rankings/mens/team-rankings/test",
"https://www.icc-cricket.com/rankings/mens/team-rankings/odi",
"https://www.icc-cricket.com/rankings/mens/team-rankings/t20i"]
tables = list()
for url in links:
    html = requests.get(url).content 
    soup = BeautifulSoup(html,"lxml") 
    temp_df = pd.read_html(str(soup))[0]
    tables.append(temp_df)
    