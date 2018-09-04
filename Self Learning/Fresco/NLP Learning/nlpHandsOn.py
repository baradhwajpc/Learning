# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 09:03:49 2018

@author: baradhwaj
"""

'''
Task 1
Import brown corpus
Determine the frequency of all words (converted into lower case) occurring in different genre of brown corpus. Store the result in brown_cfd.
Hint: Compute the condition frequency with condition being genre and event being word.
Print the frequency of modal words ['can', 'could', 'may', 'might', 'must', 'will'], in text collections associated with genre news, religion and romance, in form a of a table.
Hint : Make use of tabulate method associated with a conditional frequency distribution.
Instructions
Open app.py file using vim editor vim app.py
Press i for inserting content into file.
Write the required python code.
Save the file by pressing Esc key and typing :wq in editor.
Execute the script with command python3 app.py | tee .output.txt
Click on Continue to move to next Task.
'''

import nltk
from nltk.corpus import brown
brown_cfd = nltk.ConditionalFreqDist([ 
(genre, word) 
for genre in brown.categories() 
for word in [word.lower() for word in brown.words(categories=genre)]])
brown_cfd.tabulate(conditions=['news', 'religion' , 'romance'], 
                   samples=['can', 'could', 'may', 'might', 'must', 'will'])

'''
Task 2
Import inaugural corpus
For each of the inaugural address text available in the corpus, perform the following.
Convert all words into lower case.
Then determine the number of words starting with america or citizen.
Hint : Compute conditional frequency distribution, where condition is the year in which the inaugural address was delivered and event is either america or citizen. Store the conditional frequency distribution in variable ac_cfd.

Print the frequency of words ['america', 'citizen'] in year [1841, 1993].

Hint: Make use of tabulate method associated with a conditional frequency distribution.
'''