###                Hackathon for Machine Learning Engineer Program
###        Web Scrapping, Data Cleaning & Exploratory Analysis with Python

# In this challenge you will be scrapping the data related to cryptocurrencies,
# perform necessary data cleaning and build few visualization plots.

# NOTE: Use this link 'https://coinmarketcap.com/' as a reference to understand
# the page structure of html page used in this exercise. Both are similar.
# This reference is required for identifying required html tags for parsing the
# given web page and extracting the needed information

# Please complete the definition of following 9 functions, inorder to complete the exercise:
# 1. parse_html_page,
# 2. get_all_tr_elements,
# 3. convert_tr_elements_to_cryptodf,
# 4. transform_cryptodf,
# 5. draw_barplot_top10_cryptocurrencies_with_highest_market_value,
# 6. draw_scatterplot_trend_of_price_with_market_value
# 7. draw_scatterplot_trend_of_price_with_volum
# 8. draw_barplot_top10_cryptocurrencies_with_highest_positive_change
# 9. serialize_plot

# The above nine functions are used in 'main' function.
# The 'main' function parses an input html page, converts required info into pandas datarame,
# and draws two barplots and two scatter plots.

# Please look into definition of 'main' function, to understand inputs and exepcted outputs from above listed 9 functions.
# Also read the documention provided in each function to understand it's functionality.

## Importing python libraries, necessary for solving this exercise.
from requests import get
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import re
import pickle


def parse_html_page(htmlpage):
    '''
    Parses the input 'htmlpage' using Beautiful Soup html parser and returns it.
    '''
    # write the functionality of 'parse_html_page' below.
    soup = BeautifulSoup(open(htmlpage,encoding='utf-8'), "html.parser")
    #r = get(htmlpage)
    #soup = BeautifulSoup(r.content)
    return soup
def get_all_tr_elements(soup_obj):
    '''
    Identifies so 'tr' elements, present in beautiful soup object, 'soup_obj', and returns them
    '''
    tr_Elements = []
    # write your functionality below
    for i in soup_obj.find_all('tr'): # list element
        tr_Elements.append(i)
    return tr_Elements
def convert_tr_elements_to_cryptodf(htmltable_rows):
    '''
    Extracts the text associated with seven columns of all records present in 'htmltable_rows' object.
    Builds a pandas dataframe and returns it.

    NOTE: Information in seven columns have to be stored in below initilaized lists.
    '''
    rank = []                    #List for rank of the currency (first column in the webpage)
    currency_name = []           #List for name of the currency
    market_cap = []              #List for market cap
    price = []                   #List for price of the crypto currency
    volume = []                  #List for Volume(24h)
    supply = []                  #List for Circulating supply
    change = []                  #List for Change(24h)

    # write your functionality below
    for row in htmltable_rows[1:]:
        rank.append(row.findAll('td')[0].get_text().strip(" "))
        cName = row.findAll('td')[1].get_text().strip(" ").splitlines()
        currency_name.append(cName[-1])
        market_cap.append(row.findAll('td')[2].get_text().strip(" "))
        price.append(row.findAll('td')[3].get_text().strip(" "))
        volume.append(row.findAll('td')[4].get_text().strip(" "))
        supply.append(row.findAll('td')[5].get_text().strip(" "))
        change.append(row.findAll('td')[6].get_text().strip(" "))

    # Creating the pandas dataframe
    df = pd.DataFrame({
                         'rank' : rank,
                         'currency_name' : currency_name,
                         'market_cap' : market_cap,
                         'price' : price,
                         'volume' : volume,
                         'supply' : supply,
                         'change' : change
                         })

    # Returning the data frame.
    return df


def transform_cryptodf(cryptodf):
    '''
    Returns a modified dataframe.
    '''
    # Modify values of the columns : 'market_cap', 'price', 'volume', 'change', 'supply'.
    # Remove unwanted characters like dollar symbol ($), comma symbol (,) and
    # convert them into corresponding data type ie. int or float.

    # NOTE : After transformation, the first five rows of the transformed dataframe should be
    #       similar to data in 'expected_transformed_df.csv' file, present in project folder
    # write your functionality below
    
    newv =[]
    for v in cryptodf['volume']:
        newv.append(re.sub('[^0-9]', '', v))
    cryptodf['volume'] = newv
    
    newc =[]
    for v in cryptodf['change']:
        v = re.sub('[%]', '', v)
        newV = float(v)
        newc.append(newV)
    cryptodf['change'] = newc
    
    newmc =[]
    for v in cryptodf['market_cap']:
        newmc.append(re.sub('[^0-9]', '', v))
    cryptodf['market_cap'] = newmc
    
    newpc =[]
    for v in cryptodf['price']:
        v = re.sub('[$,]', '', v)
        newV = float(v)
        newpc.append(newV)
    cryptodf['price'] = newpc

    news =[]
    for v in cryptodf['supply']:
        news.append(re.sub('[^0-9]', '', v))
    cryptodf['supply'] = news
    
    cryptodf['market_cap'] = pd.to_numeric(cryptodf['market_cap'])
    cryptodf['supply'] = pd.to_numeric(cryptodf['supply'])
    cryptodf['volume'] = pd.to_numeric(cryptodf['volume'])

    return cryptodf
def draw_barplot_top10_cryptocurrencies_with_highest_market_value(cryptocur_df):
    '''
    Returns barplot
    '''
    # Create a horizontal bar plot using seaborn showing
    # Top 10 Cryptocurrencies with highest Market Capital.
    # Order the cryptocurrencies in Descending order. i.e
    # bar corresponding to cryptocurrency with highest market value must be on top of bar plot.
    # Write the functionality below
    
    barPlotVals = cryptocur_df.sort_values('market_cap', ascending=False).head(10)
    ax = sns.barplot(x='market_cap', y='currency_name', data=barPlotVals)
    return ax

def draw_scatterplot_trend_of_price_with_market_value(cryptocur_df):
    '''
    Returns a scatter plot
    '''
    # Create a scatter plot, using seaborn, showing trend of Price with Market Capital.
    # Consider 50 Cryptocurrencies, with higest Market Value for seeing the trend.
    # Set the plot size to 10 inches in width and 2 inches in height respectively.
    # Write the functionality below
    
    plotDf =cryptocur_df.sort_values('volume', ascending=False).head(50)[['market_cap','price']]
    ax = sns.pairplot(plotDf, x_vars=['market_cap'], y_vars=['price'])
    ax.fig.set_size_inches(10,2)
    print(ax.data.values)
    return ax
    

def draw_scatterplot_trend_of_price_with_volume(cryptocur_df):
    '''
    Returns a scatter plot
    '''
    # Create a scatter plot, using seaborn, showing trend of Price with 24 hours Volume.
    # Consider all 100 Cryptocurrencies for seeing the trend.
    # Set the plot size to 10 inches in width and 2 inches in height respectively.

    # Write the functionality below
    plotDF =cryptocur_df.sort_values('market_cap', ascending=False)[['volume','price']]
    ax = sns.pairplot(plotDF, x_vars=['volume'], y_vars=['price'])
    ax.fig.set_size_inches(10,2)
    return ax
    


def draw_barplot_top10_cryptocurrencies_with_highest_positive_change(cryptocur_df):
    '''
    Returns a bar plot
    '''
    # Create a horizontal bar plot using seaborn showing
    # Top 10 Cryptocurrencies with positive change in last 24 hours.
    # Order the cryptocurrencies in Descending order. i.e
    # bar corresponding to cryptocurrency with highest positive change must be on top of bar plot.
    # Write the functionality below
    
    barPlotVals = cryptocur_df.sort_values('change', ascending=False).head(10)
    ax = sns.barplot(x='change', y='currency_name', data=barPlotVals)
    al = list(ax.get_xticklabels())
    return ax

def serialize_plot(plot, plot_dump_file):
    '''
    Dumps the 'plot' object in to 'plot_dump_file' using pickle.
    '''
    # Write the functionality below
    pickle.dump(plot,open(plot_dump_file,'wb'))


def main():

    input_html = 'data/input_html/Cryptocurrency Market Capitalizations _ CoinMarketCap.html'

    html_soup = parse_html_page(input_html)

    crypto_containers = get_all_tr_elements(html_soup)

    crypto_df = convert_tr_elements_to_cryptodf(crypto_containers)

    crypto_df = transform_cryptodf(crypto_df)

    plot1 = draw_barplot_top10_cryptocurrencies_with_highest_market_value(crypto_df)

    plot2 = draw_scatterplot_trend_of_price_with_market_value(crypto_df)

    plot3 = draw_scatterplot_trend_of_price_with_volume(crypto_df)

    plot4 = draw_barplot_top10_cryptocurrencies_with_highest_positive_change(crypto_df)

    serialize_plot(plot1, "plot1.pk")

    serialize_plot(plot2.axes, "plot2_axes.pk")

    serialize_plot(plot2.data, "plot2_data.pk")

    serialize_plot(plot3.axes, "plot3_axes.pk")

    serialize_plot(plot3.data, "plot3_data.pk")

    serialize_plot(plot4, "plot4.pk")


if __name__ == '__main__':
    main()
