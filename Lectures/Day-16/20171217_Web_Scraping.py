from bs4 import BeautifulSoup
import pandas as pd
import requests

###### Retrieving tags from html web pages ############################
r = requests.get('http://www.mationalytics.com/sample.html')
soup = BeautifulSoup(r.content)

for i in soup.find_all('li'): # list element
    print("\nText in List item:",i.text)
    
for i in soup.find_all('a'): # hyperlinks
    print("\nText in List item:",i['href'])
    
for i in soup.find_all('img'): # images
    print("\nText in List item:",i['src'])

######## Retrieving tables in html pages ################################

links = ["https://www.icc-cricket.com/rankings/mens/team-rankings/test",
         "https://www.icc-cricket.com/rankings/mens/team-rankings/odi",
         "https://www.icc-cricket.com/rankings/mens/team-rankings/t20i"]

tables = list()

for url in links:
    
    html = requests.get(url).content

    soup = BeautifulSoup(html,"lxml")

    temp_df = pd.read_html(str(soup))[0]
    
    tables.append(temp_df)

test_df = tables[0].iloc[:,[1,3]]
odi_df = tables[1].iloc[:,[1,3]]
t20_df = tables[2].iloc[:,[1,3]]

## Assignment 1
# inner merge all the three tables and find sum of points across
# all format and find highest 

## Assignment 2

# url : http://www.espncricinfo.com/wi/engine/series/1078425.html?view=pointstable

# Sum the points of each team in the points by match table and 
# create a dataframe with columns 'Team' and "Total points" and compare it with Points Table
