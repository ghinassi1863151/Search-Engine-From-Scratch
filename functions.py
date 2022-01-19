#############################################################################    
############################## IMPORTS ######################################
#############################################################################

# Import parsing libraries
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import os
from dateutil.parser import parse
from datetime import datetime
import time
# Import folder libraries
import os
import csv
import re
from shutil import copyfile
# Import preprocessing libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import SnowballStemmer, PorterStemmer
import string
# Import queries libraries
from functools import reduce
import numpy as np
import pandas as pd
import heapq
# Import libraries to save and load
import pickle
import json as js
# Set up
import glob
from collections import Counter

#function for save and load in python/json objects the dictionaries
def save_pickle(dic, path):
    with open(f"{path}", 'wb') as f:
        pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    with open(f"{path}", 'rb',) as f:
        return pickle.load(f)

#############################################################################    
########################## GLOBAL VARIABLES #################################
#############################################################################

header = ['animeTitle', 'animeType', 'animeNumEpisode', 'animeRelDate', 'animeEndDate', 
                        'animeMembers', 'animeScore', 'animeUsers', 'animeRank', 'animePopularity', 'animeSynopsis',
                        'animeRelated', 'animeCharacters', 'animeVoices', 'animeStaff']

D = load_pickle("./D.pickle")
preproc_D = load_pickle("./PP_D.pickle")
inv_index = load_pickle("./inv_index.pickle")
vocab = load_pickle("./vocabulary.pickle")
index = load_pickle("./tfidf_D.pickle")
preproc_titles = load_pickle("./preproc_titles.pickle")
titles = load_pickle("./titles.pickle")
titles_dict = load_pickle("./titles_dict.pickle")
popularity = load_pickle("/home/arturo/ADM/HW3/score.pickle")

#############################################################################    
########################## PARSING FUNCTION #################################
#############################################################################

# creates a list with the element that differ between two lists
def list_diff(l1, l2):
    l_diff = [i for i in l1 + l2 if i not in l1 or i not in l2]
    return l_diff


# 1 - Name
def animeTitle(html_string):
    with open(html_string, 'r') as f:
      soup = BeautifulSoup(f, 'html.parser')
      box = soup.find_all('h1', {'class':'title-name h1_bold_none'})
      for i in box:
        animeTitle = i.find('strong').contents[0].strip()

    return animeTitle


# 2 - Type 
def animeType(html_string):
    with open(html_string, 'r') as f:
      soup = BeautifulSoup(f, 'html.parser') 
      animeType = soup.select_one("a[href*=type]").contents[0].strip()
      
    return animeType

# 3 - Number of episodes
def animeNumEpisode(html_string):
    with open(html_string, 'r') as f:
      soup = BeautifulSoup(f, 'html.parser')
      try:
        numEpisode = int(soup.find(text=re.compile('Episodes:')).parent.parent.text.split()[-1])
      except ValueError:
        numEpisode = ''

    return numEpisode

# 4 - Release and End dates
def animeRelDate(html_string):
    with open(html_string, 'r') as f:
      soup = BeautifulSoup(f, 'html.parser')
      dates = soup.find(text=re.compile('Aired:'), class_="dark_text").parent.text.split()[1:] 
      # the output of date is ['M', 'D', 'Y', 'to', 'M', 'D', 'Y'] where only the release Year is always present if the date is available

      if ' '.join(dates) == 'Not available': 
           releaseDate = ''
      elif len(dates) == 7:                                         # ['M', 'D', 'Y', 'to', 'M', 'D', 'Y']
           releaseDate = parse(' '.join(dates[0:3])).date()
      elif len(dates)==6:  
           if dates[2]=="to":                                       # ['M', 'Y', 'to', 'M', 'D', 'Y']
              releaseDate = parse(''.join(dates[1])).date().year
           else:                                                    # ['M', 'D', 'Y', 'to', 'M', 'Y']
              releaseDate = parse(' '.join(dates[:2])).date()
      elif len(dates) == 5:
           if dates[1] == 'to':                                     # ['Y', 'to', 'M', 'D', 'Y']
              releaseDate = parse(''.join(dates[0])).date().year
           elif dates[3] == 'to':                                   # ['M', 'D', 'Y', 'to', 'Y']
              releaseDate = parse(' '.join(dates[:3])).date()
           else:                                                    # ['M', 'Y', 'to', 'M', 'Y']
              releaseDate = parse(''.join(dates[1])).date().year
      elif len(dates) == 4:
           if dates[2] == 'to':                                     # ['M', 'Y', 'to', 'Y']
              releaseDate = parse(''.join(dates[1])).date().year
           else:                                                    # ['Y', 'to', 'M', 'Y']
              releaseDate = parse(''.join(dates[0])).date().year
      elif len(dates) == 3:
          if dates[1] == 'to':                                      # ['Y', 'to', 'Y']
            releaseDate = parse(''.join(dates[0])).date().year
          else:                                                     # ['M', 'D', 'Y']
            releaseDate = parse(' '.join(dates[:3])).date()
      elif len(dates) == 2:                                         # ['M', 'Y']
          releaseDate = parse(''.join(dates[1])).date().year
      else:                                                         # ['Y']
        releaseDate = parse(''.join(dates[0])).date().year

    return releaseDate

def animeEndDate(html_string):
    with open(html_string, 'r') as f:
      soup = BeautifulSoup(f, 'html.parser')
      dates = soup.find(text=re.compile('Aired:'), class_="dark_text").parent.text.split()[1:]

      if len(dates) == 7:
         endDate = parse(' '.join(dates[4:])).date()
      elif len(dates)==6:
         if dates[2]=="to":
            endDate = parse(' '.join(dates[3:])).date()
         else: 
            endDate = parse(''.join(dates[5])).date().year
      elif len(dates) == 5:
         if dates[4] == '?':
            endDate = ''
         else:
            endDate = parse(''.join(dates[4])).date().year
      elif len(dates) == 4:
         if dates[3] == '?':
            endDate = ''
         else:
            endDate = parse(''.join(dates[3])).date().year
      elif len(dates) == 3:
         if dates[1] == 'to' and dates[2] != '?': # ['Y', 'to', 'Y']
            endDate = parse(''.join(dates[2])).date().year
         else:
            endDate = ''
      else:
         endDate = ''
        
    return endDate

# 5 - Number of members
def animeMembers(html_string):
    with open(html_string,'r') as f:
      soup = BeautifulSoup(f, 'html.parser')
      box = soup.find('span', {'class':'numbers members'})
      animeMembers = int(box.find('strong').contents[0].replace(',', '').strip())
      
      return animeMembers

# 6 - Score
def animeScore(html_string):
    with open(html_string, 'r') as f:
      soup = BeautifulSoup(f, 'html.parser')
      for i in range(10):
        score = soup.find("div", {'class':'score-label score-'+str(i)})
        if score != None:
           animeScore = float(score.text)
        else:
           animeScore = ''

    return animeScore

# 7 - Users
def animeUsers(html_string):
    with open(html_string, 'r') as f:
      soup = BeautifulSoup(f, 'html.parser')
      if soup.find(itemprop="ratingCount") != None:
        animeUsers = int(soup.find(itemprop="ratingCount").get_text()) 
      else:
        animeUsers = ''

    return animeUsers

# 8 - Rank
def animeRank(html_string):
    with open(html_string,'r') as f:
      soup = BeautifulSoup(f, 'html.parser')
      box = soup.find('span', {'class':'numbers ranked'})
      if box.find('strong').contents[0].strip() != 'N/A':
        animeRank = int(box.find('strong').contents[0].strip()[1:])
      else:
        animeRank = ''
      
      return animeRank

# 9 - Popularity
def animePopularity(html_string):
  with open(html_string,'r') as f:
    soup = BeautifulSoup(f, 'html.parser')
    box = soup.find('span', {'class':'numbers popularity'})
    animePop = int(box.find('strong').contents[0].strip()[1:])

    return animePop
 

# 10 - Synopsis
def animeSynopsis(html_string): 
  with open(html_string, 'r') as f:
    soup = BeautifulSoup(f, 'html.parser')
    syn = soup.find(itemprop="description")
    animeSynopsis = ' '.join(syn.get_text().split())

  return animeSynopsis

# 11 - Related Anime
def animeRelated(html_string):  
    animeRelated = []
    with open(html_string, 'r') as f:
      soup = BeautifulSoup(f, 'html.parser')
      related = soup.find_all("table", {"class":"anime_detail_related_anime"})
      for i in related:
        links = i.find_all('a')
        for link in links:  
            try:
               animeRelated.append(f'{link.contents[0]}'.strip())
            except IndexError:
               pass
            continue
      if len(animeRelated) == 0: animeRelated = ''

    return animeRelated


# 12 - Characters
def animeCharacters(html_string):

    animeCharacters = []
    with open(html_string, 'r') as f:
      soup = BeautifulSoup(f, 'html.parser')
      for tag in soup.find_all('h3'):
        links = tag.find_all('a')
        for link in links:        
            animeCharacters.append(f'{link.contents[0]}'.strip())

    animeCharacters = animeCharacters[:-3]
    if len(animeCharacters)==0: animeCharacters = ''
    
    return animeCharacters

# 13 - Voices
def animeVoices(html_string):
  
  animeVoices = []  
  with open(html_string, 'r') as f:
    soup = BeautifulSoup(f, 'html.parser')
    voices = soup.find_all("td", {"class":"va-t ar pl4 pr4"})
    for i in voices:
        bucket = i.select("a[href*=people]")
        for el in bucket:
          animeVoices.append(el.contents[0].strip())
  
  return animeVoices


# 14 - Staff
def animeStaff(html_string):
  
  animeStaff = [] 
  task = [] 
  with open(html_string, 'r') as f:
    soup = BeautifulSoup(f, 'html.parser')
    staff = soup.find_all("td", {"class":"borderClass"})
    for i in staff:
        bucket = i.select("a[href*=people]")
        for el in bucket:
            animeStaff.append(el.contents[0].strip())

    staff_members = [i.strip() for i in list(filter(('\n').__ne__, list_diff(list(filter(('').__ne__, animeStaff)), animeVoices(html_string))))]
    
    for i in staff:
        bucket = i.find_all("div", {"class":"spaceit_pad"})
        for el in bucket:
            a = el.select("small")
            for x in a:
              task.append(x.get_text())
   
    task = task[:-(len(staff_members)+1):-1]
    task = [i.strip() for i in task]
    task.reverse()
    
    animeStaff = [list(x) for x in zip(staff_members, task)]
    if len(animeStaff) == 0: animeStaff = ''

  return animeStaff

#functon for the parsing using all the previus functions
def html_parsing(filename, h):
    # h is the html path given as a string
    with open('/content/drive/MyDrive/ADM-HW3/'+filename, 'wt') as out_file:
        anime_tsv = csv.writer(out_file, delimiter='\t')
        anime_tsv.writerow([animeTitle(h), animeType(h), animeNumEpisode(h), animeRelDate(h), animeEndDate(h), 
                            animeMembers(h), animeScore(h), animeUsers(h), animeRank(h), animePopularity(h), animeSynopsis(h),
                            animeRelated(h), animeCharacters(h), animeVoices(h), animeStaff(h)])
        
        
        
#############################################################################    
############################# SEARCH ENGINE #################################
#############################################################################

#creates a dictionary starting from a tsv
def Dict(file_path):
    d={}
    with open(f"{file_path}", 'r', encoding='utf-8') as f:
      tsv = f.read().split('\t')
      d = {k: v for k,v in zip(header, tsv)}

    return d

# we create a function where we execute all the preprocessing steps, we add the procedures as parameters
# so we can use the same function to do different preprocessing depending from the variable

def text_preprocessing(var, lower=False, numbers=False, stemming=False):
    stop_words = set(stopwords.words("english"))

    # converting to lowercase
    if lower == True:
        var = var.lower()

    # removing punctuation
    var.translate(str.maketrans('', '', string.punctuation))

    # tokenization
    tokens = word_tokenize(var)

    # removing stopwords
    filtered_text = []
    for each_word in tokens:
        if each_word not in stop_words:
            filtered_text.append(each_word)

    # removing numbers
    if numbers == True:
        filtered_text = [token for token in filtered_text if not token.isdigit()]

    # stemming
    if stemming == True:
        stemmer = PorterStemmer()
        filtered_text = [stemmer.stem(token) for token in filtered_text]

    # join tokens
    out = " ".join(filtered_text)

    return out, filtered_text

# given a value of a dictionary returnes the corresponding key
def get_key(D, val):
    for key, value in D.items():
         if val == value:
             return key
 
    return "key doesn't exist"

# given a list of token and a list of documents it computes the inverted index
def inverted_index(l1, l2):
    inv_index = {}

    for val in tqdm(l1):
        docs = []
        for i in range(0, len(l2)):
            if val in l2[i]:
                docs.append(f'anime_{i + 1}')

        inv_index[val] = docs

    return inv_index

# an interactive search bar for the queries
def search():

    query, tkn = text_preprocessing(input('search bar: '), lower=True, numbers=True, stemming=True)

    docs = []
    try:
        for i in tkn:
            docs.append(set(inv_index[i]))
            
        result = reduce(set.intersection, docs)
        if len(result) == 0:
            result = reduce(set.union, docs)

        titles, synopsis, urls = [], [], []
        for res in result:
            titles.append(D[res]['animeTitle'])
            synopsis.append(D[res]['animeSynopsis'])
            urls.append(D[res]['animeUrl'][0])

        out = {'animeTitle': titles, 'animeDescription': synopsis, 'Url': urls}
        df = pd.DataFrame(out, index=range(1, len(result) + 1))
        
    except:

        print('\nNo solution found, maybe the word you are searching is a stopword. \nTry to add more informations')
        return None

    return df

# an interactive search bar for the queries that also computes the similarity between the queries and the synopsis
def search_cosine():

    query, tkn = text_preprocessing(input('search bar: '), lower=True, numbers=True, stemming=True)

    docs = []
    try:
        for i in tkn:
            docs.append(index[i])

        q = {}
        for i in range(len(tkn)):
            tfidf = {}
            for j in range(len(docs[i])):
                anime_id, tfidf_anime_i_token_j = docs[i][j][0], docs[i][j][1]
                
                tfidf[anime_id] = tfidf_anime_i_token_j
                docs[i][j] = anime_id

            q[tkn[i]] = tfidf
            docs[i] = set(docs[i])

        result = reduce(set.intersection, docs)

        if len(result) == 0:
            result = reduce(set.union, docs)

        sim_dict = {}
        for res in result:
            cosine = 0
            d = preproc_D[res]['animeSynopsis']
            for t in tkn:
                try:
                    cosine += q[t][res]  
                    # when we are in the set.union case it could happen that we don't have a certain 
                    # anime for a certain token so we penalize the cosine by adding 0
                except KeyError:
                    cosine += 0
                continue

            sim_dict[res] = round((cosine / len(d)) * 100, 2)

        heap = [(-value, key) for key, value in sim_dict.items()]
        largest = heapq.nsmallest(3, heap)
        largest = [(key, -value) for value, key in largest]

        titles, synopsis, urls, similarity = [], [], [], []

        for x in largest:
            name = x[0]
            titles.append(D[name]['animeTitle'])
            synopsis.append(D[name]['animeSynopsis'])
            urls.append(preproc_D[name]['animeUrl'][0])
            similarity.append(x[1])

        out = {'animeTitle': titles, 'animeDescription': synopsis, 'Url': urls, 'Similarity': similarity}
        df = pd.DataFrame(out, index=range(1, len(largest) + 1))

    except:
        print('\nNo solution found, maybe the word you are searching is a stopword. \nTry to add more informations')
        return None

    return df
