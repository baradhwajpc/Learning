# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 17:52:00 2018

@author: baradhwaj
"""
from bs4 import BeautifulSoup
import pandas as pd
import requests
import re
import seaborn as sns


soup = BeautifulSoup(open('G:\\new.html',encoding='utf-8'), "html.parser")
BeautifulSoup(open('C:\Users\Soumadeep_Bhattacha\Downloads\xyz.html',encoding='utf-8'), html.parser)
#r = requests.get('https://coinmarketcap.com/')
#soup = BeautifulSoup(r.content,"lxml")
sp_Obj = []

for i in soup.find_all('tr'): # list element
    #print("\nText in List item:",i.text)
    sp_Obj.append(i)    

rank = []                    #List for rank of the currency (first column in the webpage)    
currency_name = []           #List for name of the currency
market_cap = []              #List for market cap
price = []                   #List for price of the crypto currency
volume = []                  #List for Volume(24h)
supply = []                  #List for Circulating supply
change = []                  #List for Change(24h)

for row in sp_Obj[1:]:
    rank.append(row.findAll('td')[0].get_text().strip(" ").replace('\n', ''))
    cName = row.findAll('td')[1].get_text().strip(" ").splitlines()
    currency_name.append(cName[-1])
    market_cap.append(row.findAll('td')[2].get_text().strip(" ").replace('\n', ''))
    price.append(row.findAll('td')[3].get_text().strip(" ").replace('\n', ''))
    volume.append(row.findAll('td')[4].get_text().strip(" ").replace('\n', ''))
    supply.append(row.findAll('td')[5].get_text().strip(" ").replace('\n', ''))
    change.append(row.findAll('td')[6].get_text().strip(" ").replace('\n', ''))
    
    df = pd.DataFrame({
                         'rank' : rank,
                         'currency_name' : currency_name,
                         'market_cap' : market_cap,
                         'price' : price,
                         'volume' : volume,
                         'supply' : supply,
                         'change' : change
                         })
    
     
    newv =[]
    for v in df['volume']:
        newv.append(re.sub('[^0-9]', '', v))
    df['volume'] = newv
    

    newc =[]
    for v in df['change']:
        v = re.sub('[%]', '', v)
        newV = float(v)
        newc.append(newV)
    df['change'] = newc
    

    newmc =[]
    for v in df['market_cap']:
        newmc.append(re.sub('[^0-9]', '', v))
    df['market_cap'] = newmc
    
    newpc =[]
    for v in df['price']:
        v = re.sub('[$,]', '', v)
        newV = float(v)
        newpc.append(newV)
    df['price'] = newpc

    news =[]
    for v in df['supply']:
        news.append(re.sub('[^0-9]', '', v))
    df['supply'] = news




    #cryptodf.convert_objects(convert_numeric=True)
    df['market_cap'] = pd.to_numeric(df['market_cap'])
    df['supply'] = pd.to_numeric(df['supply'])
    df['volume'] = pd.to_numeric(df['volume'])

    
    
    
    ''' Plots '''
    
    barPlotVals = df.sort_values('market_cap', ascending=False).head(10)
    sns.barplot(x='market_cap', y='currency_name', data=barPlotVals, orient='h')
    
    sns.set_color_codes("muted")
    colors = ['red' if (x < 20) else 'green' for x in df['market_cap']]
    sns.barplot(data=barPlotVals, x='market_cap', y='currency_name', palette=colors)

    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    
    scatterPlotVals = df.sort_values('market_cap', ascending=False).head(50)
    ax = sns.pairplot(x='price', y='market_cap', data=scatterPlotVals,
           fit_reg=False)
    
    ax = sns.lmplot(x='price', y='market_cap', data=scatterPlotVals,
           fit_reg=False)
    
    
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.ticklabel_format(style='plain', axis='y')
    ax.get_xticklabels()
    ax.fig.set_size_inches(10,2)

    #, # No regression line           hue='Legendary')   # Co
    
    
    scatterPlotVals = df.sort_values('market_cap', ascending=False).head(50)
    ax = sns.regplot(x='price', y='market_cap', data=scatterPlotVals,
           fit_reg=False)
    ax.set(xticks=df.a[2::8])

    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.fig.set_size_inches(10,2)

    
    ax = sns.scatterplot(x="price", y="market_cap",
                      data=scatterPlotVals)
    
    
    barPlotVals = df.sort_values('change', ascending=False).head(10)
    ax = sns.barplot(data=barPlotVals, x='change', y='currency_name')
    ax.get_xaxis().get_major_formatter().set_scientific(False)

    return ax
    
    
    barPlotVals = df.sort_values('change', ascending=False).head(10)
    ax2 = sns.barplot(data=barPlotVals, x='change', y='currency_name')
    return ax2
    
    scatterPlotVals = df.sort_values('market_cap', ascending=False).head(50)
    #ax = sns.lmplot(y='price', x='market_cap', data=scatterPlotVals,           fit_reg=False)
    ax = sns.pairplot(data=scatterPlotVals[["price","market_cap"]])
    ax.fig.set_size_inches(10,2)
    
    
    ax = sns.pairplot(data=df[["price","volume"]])
    ax.fig.set_size_inches(10,2)
   
    barPlotVals = df.sort_values('change', ascending=False).head(10)
    ax4 = sns.barplot(x='change', y='currency_name', data=barPlotVals)
    ax4.get_xaxis().get_major_formatter().set_scientific(False)
    ax4.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    al = list(ax4.get_xticklabels())
    for l in al:
        t= l
        try:
            print(t.get_text())
            t.set_text("{:.2f}".format(float(t.get_text())))
        except ValueError :
            print('1')
            #print "error",e,"on line",i
        l = t
        #print(t.text)
    ax4.set_xticklabels(al, fontdict=None, minor=False)
    
    plotDF =df.sort_values('market_cap', ascending=False)[['volume','price']]
    ax = sns.pairplot(plotDF, x_vars=['volume'], y_vars=['price'])
    ax.fig.set_size_inches(10,2)
    
    
    aab ='Text(0.4,0,"0.4")'
    print([x for x in pattern.split(aab) if x])