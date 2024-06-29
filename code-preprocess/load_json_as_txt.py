#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:21:56 2021

@author: korneliabastin
"""

import json
import nltk
import pandas as pd
import numpy as np
import statistics
import string
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer




#  Get input JSON

if len( sys.argv ) != 2:
    print( 'Usage:  python load_json_as_txt.py INPUT_JSON_FILENAMNE' )
    sys.exit( 0 )

#with open("1_atlanta.json") as f:

#  Read user-specified JSON into data

with open( sys.argv[ 1 ] ) as f:
    data = json.loads(f.read())
    text = data["results"]["transcripts"][0]["transcript"]

"""      
text = text.lower()   
text_old = text
text = text[text.find('green'):]
"""


import json
import re


"""
person_string = '{"name": "Bob", "languages": "English", "numbers": [2, 1.6, null]}'

# Getting dictionary
person_dict = json.loads(person_string)

# Pretty Printing JSON string back
print(json.dumps(data, indent = 4, sort_keys=True))
"""


#  Pull results as individual items from JSON

new_data = data["results"]["items"]


"""
for term in nltk.word_tokenize( post ):
        if term not in stop_w:
            #t_vec.append( snowball_stem.stem( term ) )
            if term == 'full': term = 'fuel'
            t_vec.append(term)
            #t_vec2.append(term)
    #print(t_vec)
    #s[0][ i ] = ' '.join( t_vec )
    text2[ i ] = ' '.join(t_vec)
    #text_all[ i ] = ' '.join(t_vec2)
"""

data1 = 1000 * [0]
j = 0
subcluster = []
start_time = []


for i in range(len(new_data)):

    #  Handle start of pronunciation type

    if new_data[i]["type"] == "pronunciation" and i == 0:
        subcluster.append(new_data[i]["alternatives"][0]["content"])
        start_time.append(new_data[i]["start_time"])

    #  Handle continuation of pronunciation type
    
    elif new_data[i]["type"] == "pronunciation" and i != 0:
        if (
            new_data[i - 1]["type"] == "pronunciation"
            and float(new_data[i - 1]["end_time"])
            >= float(new_data[i]["start_time"]) - 5
        ):

            #  More than 5 sec have passed so this is a new speech chunk
            
            subcluster.append(new_data[i]["alternatives"][0]["content"])

        #  Previous data type was punctuation?
        
        elif new_data[i - 1]["type"] == "punctuation":

            #  New speech chunk (more than 5 sec passed)? Then create new chunk
            
            if (
                float(new_data[i - 2]["end_time"])
                >= float(new_data[i]["start_time"]) - 5
            ):
                subcluster.append(new_data[i]["alternatives"][0]["content"])

            #  Otherwise append to current speech chunk
            
            else:
                data1[j] = " ".join(subcluster)
                subcluster = []
                j += 1
                subcluster.append(new_data[i]["alternatives"][0]["content"])
                start_time.append(new_data[i]["start_time"])

        #  Pronunciation and less than 5 secs have passed, append to current chunk

        else:
            data1[j] = " ".join(subcluster)
            subcluster = []
            j += 1
            subcluster.append(new_data[i]["alternatives"][0]["content"])
            start_time.append(new_data[i]["start_time"])

    #  Otherwise fallback to start a new speech chunk
    
    elif new_data[i]["type"] == "punctuation":
        subcluster.append(new_data[i]["alternatives"][0]["content"])

#  Add speech chunk to end of data1

data1[-1] = " ".join(subcluster)


#  Remove first entry from data1

data1 = [i for i in data1 if i != 0]

#  Remove specific punctuation from end of strings in data1

for i in range(len(data1)):
    data1[i] = re.sub(r'\s+([?.!",])', r"\1", data1[i])


# text2 =  text.split(".")

text2 = data1

#  Look for first green flag tag

for i in range(len(text2)):
    if "green" in text2[i] or "Green" in text2[i]:
        text2 = text2[i:]
        start_time = start_time[i:]
        break

##############################################################################

from nltk import tokenize

#  Tokenize speech (as text) for each speech chunk and record start time
#  of chunk

new_start_time = []
for i in range(len(text2)):
    text2[i] = tokenize.sent_tokenize(text2[i])
    new_start_time += [start_time[i]] * len(text2[i])

text2 = [val for sublist in text2 for val in sublist]
start_time = new_start_time.copy()

##############################################################################


#########################################################################

#  Identify co-occuring duplicates in text

duplicates = []
for i in range(len(text2)):
    if i > 0:
        if text2[i - 1] == text2[i]:
            duplicates.append(i)

duplicates.reverse()

#  Remove duplicates

for i in duplicates:
    start_time.pop(i)
    text2.pop(i)


#  Combine any text less than 30 chars

for j in range(len(text2)):
    for i in range(len(text2)):
        if i >= 0 and i < len(text2) and len(text2[i]) < 30:
            text2[i - 1] = text2[i - 1] + text2[i]
            text2.pop(i)
            start_time.pop(i)
            if i >= len(text2):
                break
    if j >= len(text2):
        break


#  Remove punctuation and stop words, including NASCAR-specific
#  stop words

punc = re.compile("[%s]" % re.escape(string.punctuation))
stop_w = nltk.corpus.stopwords.words("english")

#  Additional NASCAR-specific stopwords

stop_w.extend(
    [
        "Okay",
        "okay",
        "mhm",
        "Mhm",
        "Copy",
        "10 4",
        "yes",
        "Yes",
        "yeah",
        "Yeah",
        "Clear",
        "clear",
        "copy",
        "thank you",
        "so",
        "to",
        "go",
        "uh",
        "huh",
        "sir",
        "temple",
    ]
)
snowball_stem = nltk.stem.SnowballStemmer("english")
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

# text3 = [s for s in text2 if s not in stop_w]

text_all = text2.copy()

for i, post in enumerate(text2):
    post = punc.sub(
        "", post
    ).lower()  # Remove punctuation, convert to lowercase
    # print(post)
    #  Replace original news string with no punctuation, lowercase, stop word removed, stemmed version
    t_vec = []
    t_vec2 = []
    for term in nltk.word_tokenize(post):
        if (term not in stop_w) and term in english_vocab:
            # t_vec.append( snowball_stem.stem( term ) )
            if term == "full":
                term = "fuel"
            t_vec.append(term)
            # t_vec2.append(term)
    # print(t_vec)
    # s[0][ i ] = ' '.join( t_vec )
    text2[i] = " ".join(t_vec)
    # text_all[ i ] = ' '.join(t_vec2)

#  Remove empty strings

text3 = [x for x in text2 if x != ""]
text_all = [x for x in text_all if x != ""]

#  Remove short (<=2 chars) strings

text4 = [x for x in text3 if len(x) > 2]
text_all = [x for x in text_all if len(x) > 2]

new_start_time = []
new_text_all = []
for i, x in enumerate(text_all):
    if len(x) <= 2:
        continue
    new_text_all += [x]
    new_start_time += [start_time[i]]


#  Consolidate preprocessing into single variables

text_all = new_text_all.copy()
start_time = new_start_time.copy()
#################################################

#  Important event keywords

key_words = [
    "tire",
    "save",
    "fuel",
    "rear",
    "front",
    "tight",
    "loos",
    "lose",
    "loos",
    "adjust",
    "bounc",
    "321",
    "pit",
    "balanc",
    "engin",
    "handl",
    "troubl",
    "mess",
    "issu",
    "bad",
    "entri",
    "exit",
    "bottom",
    "wedg",
    "steer",
    "break",
    "damag",
    "brace",
    "broke",
    "bust",
    "crack",
    "blown",
    "wrong",
    "fix",
    "free",
]


#  Weight terms based on known important NASCAR terms (key_words above)

text_weights = [0] * len(text2)
for i in range(len(text2)):
    for j in key_words:
        if j in text2[i]:
            # print(text2[i])
            text_weights[i] = text_weights[i] + 10

#  Create sentence dataframes

sentences = pd.DataFrame({"sentence": text2, "weight": text_weights})
sentences = pd.DataFrame(
    {"sentence": text2, "weight": text_weights, "time": start_time}
)

# y = sentences.sort_values(by=['weight'])

#  Summarize senteces as those with weight > 0

summary = sentences.loc[(sentences["weight"] > 0)]

text_new2 = pd.DataFrame({"sentence2": text_all})

new = pd.merge(summary, text_new2, left_index=True, right_index=True)

#  Update text weights

text_weights = [0] * len(text2)
for i, txt in enumerate(text2):
    for j in key_words:
        if j in txt:
            # print(text2[i])
            text_weights[i] = text_weights[i] + 10

xxx = new["sentence2"]

##############################################################################
#  Write sentences and associated weights and start times to CSV

sentences = pd.DataFrame(
    {"sentence": text_all, "weight": text_weights, "time": start_time}
    #    {"sentence": new["sentence"], "weight": text_weights, "time": new["time"]}
)
sentences.to_csv("test_3v1.csv", index=False)


##############################################################################

#  Empty any sentence with a text weight of 0

for i in range(len(text2)):
    if text_weights[i] == 0:
        text2[i] = ""

##############################################################################
##############################################################################
##############################################################################

#  Important event keywords

key_words = [
    "save",
    "fuel",
    "rear",
    "front",
    "tight",
    "loos",
    "lose",
    "loos",
    "adjust",
    "bounc",
    "321",
    "pit",
    "balanc",
    "engin",
    "handl",
    "troubl",
    "mess",
    "issu",
    "bad",
    "entri",
    "exit",
    "bottom",
    "wedg",
    "steer",
    "break",
    "damag",
    "brace",
    "broke",
    "bust",
    "crack",
    "blown",
    "wrong",
    "fix",
    "free",
]


#  For all speech chunks

for i, post in enumerate(text2):

    #  Remove punctuation, lowercase entire chunk
    
    post = punc.sub(
        "", post
    ).lower()  # Remove punctuation, convert to lowercase
    # print(post)

    #  Replace original string chunk with no punctuation, lowercase, stop word
    #  removed, stemmed version

    t_vec = []
    t_vec2 = []
    for term in nltk.word_tokenize(post):
        if (
            (term not in stop_w)
            and (term in english_vocab)
            and not (term.isnumeric())
        ):
            t_vec.append(snowball_stem.stem(term))
            # t_vec.append(term)
            # t_vec2.append(term)
    # print(t_vec)
    # s[0][ i ] = ' '.join( t_vec )

    #  Skip anything with 5 or fewer characters
    
    if len(t_vec) > 5:
        text2[i] = " ".join(t_vec)
    else:
        text2[i] = " "
    # text_all[ i ] = ' '.join(t_vec2)

#  Grab a list of empty string locations

empty = []
for i in range(len(text2)):
    if text2[i] == " ":
        empty.append(i)

empty.reverse()

#  Remove empty strings and corresponding start times

for i in empty:
    text2.pop(i)
    text_all.pop(i)
    start_time.pop(i)


# text2 = [x for x in text2 if x !=' ']
# text_all = text2.copy()

#  Use sklearn counter vectorizer to query term counts

vectorizer = CountVectorizer()

X_tf = vectorizer.fit_transform(text2)
shp = X_tf.toarray().shape
print(f"Documents: {shp[0]}; Unique Terms: {shp[1]}")

#  List unique terms

terms = vectorizer.get_feature_names_out()
terms = list(terms)

#  TF-IDF weight each speech chunk

X_tf = X_tf.todense()
X_tf = np.asarray(X_tf)

for i in key_words:
    for j in range(len(text2)):
        if i in text2[j]:
            if i in terms:
                freq = X_tf.item(j, terms.index(i))
                X_tf.itemset((j, terms.index(i)), freq * 1.5)

tfidf = TfidfTransformer()
tfidfMatrix = tfidf.fit_transform(X_tf).toarray()
print(tfidfMatrix.shape)


#  Normal TF-IDF matrix to fix for short vs. long documents (normalize over
#  term count)


x_norm = []

length = np.linalg.norm(tfidfMatrix, axis=1)  #  List of document vector lengths

for i in range(
    0, tfidfMatrix.shape[0]
):  #  For each document, normalize its vector
    x_norm.append([x / length[i] for x in tfidfMatrix[i]])

###########################Different TFIDF######################################

#  Re-run TF-IDF with a different sklearn algorithm

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(text2).toarray()

print(X_tfidf.shape)

terms = vectorizer.get_feature_names_out()
term_list = list(terms)

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

#  Re-normalize terms for each document to correct for document length

x_norm = []

length = np.linalg.norm(X_tfidf, axis=1)  #  List of document vector lengths

for i in range(0, X_tfidf.shape[0]):  #  For each document, normalize its vector
    x_norm.append([x / length[i] for x in X_tfidf[i]])

##############################################################################
##############################################################################
##############################################################################

#  Increase weights for important NASCAR terms

for i in key_words:
    for j in range(len(text2)):
        if i in text2[j]:
            if i in terms:
                X_tfidf[j][term_list.index(i)] += 0.25

##############################################################################
##############################################################################
##############################################################################

x_weight = []

for i in range(len(x_norm)):
    x_norm[i] = [x for x in x_norm[i] if x != 0.0]
    x_weight.append(statistics.mean(x_norm[i]))


#  Create weight list for each speech chunk


x_weight = []

for i in range(len(x_norm)):
    x_norm[i] = [x for x in x_norm[i] if x != 0.0]
    if x_norm[i] == [] or x_norm[i] == [1]:
        x_weight.append(0)
    else:
        x_weight.append(statistics.mode(x_norm[i]))

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

#  Create dataframe w/sentence and corresponding TF-IDF weight


sentences2 = pd.DataFrame({"sentence": text2, "weight": x_weight})


#  Create summary by sorting sentences by descending weight, only keeping
#  sentences w/weight >= 0.2, then writing summaries to a CSV file

y2 = sentences2.sort_values(by=["weight"])

summary2 = sentences2.loc[(sentences2["weight"] >= 0.2)]

text_new22 = pd.DataFrame({"sentence2": text_all})

new2 = pd.merge(summary2, text_new22, left_index=True, right_index=True)

xxx2 = new2["sentence2"]

new_test = pd.merge(sentences2, text_new22, left_index=True, right_index=True)


new_test.to_csv("test_3vtfidfv1.csv", index=False)

##############################################################################
##############################################################################
##############################################################################


# Print IDF values

df_idf = pd.DataFrame(
    tfidf.idf_, index=vectorizer.get_feature_names_out(), columns=["idf_weights"]
)

# Sort dataframe rows ascending by IDF weights

df_idf.sort_values(by=["idf_weights"])

names = vectorizer.get_feature_names_out()

sentence_weights = []

#  For every sentence chunk, modify its weight based on whether it contains
#  key words and/or words in the TF-IDF term space

for i, post in enumerate(text2):
    # post = punc.sub( '', post ).lower()    # Remove punctuation, convert to lowercase
    # print(post)
    #  Replace original news string with no punctuation, lowercase, stop word removed, stemmed version
    t_vec = 0
    number_of_terms = 0

    #  For every term in speech chunk
    
    for term in nltk.word_tokenize(post):

        #  Add term's weight if it is in list of TF-IDF terms
        
        if term in names:
            t_vec = t_vec + list(df_idf.loc[term])[0]

        #  Increase term's weight if it is in list of key event terms
        
        if term in key_words:
            t_vec = t_vec + 5

        #  Maintain running term count
        
        if term in names:
            number_of_terms = number_of_terms + 1
            # if term in extra_weight:
            # number_of_terms = number_of_terms + 1

    #  If TF-IDF or keyword terms found, normalize by total number of terms

    if number_of_terms != 0:
        # if number_of_terms == 1:
        # t_vec = 0
        # elif number_of_terms <= 3:
        # t_vec = t_vec/(number_of_terms)
        # else:
        t_vec = t_vec / number_of_terms

    # print(t_vec)
    # s[0][ i ] = ' '.join( t_vec )

    #  Update sentence weight for given speech chunk
    
    sentence_weights.append(t_vec)

#  Write results for each summarized chunk to CSV file

sentences2 = pd.DataFrame({"sentence": text2, "weight": sentence_weights})

y2 = sentences2.sort_values(by=["weight"])

summary2 = sentences2.loc[(sentences2["weight"] >= 5)]

text_new22 = pd.DataFrame({"sentence2": text_all})

new2 = pd.merge(summary2, text_new22, left_index=True, right_index=True)

xxx2 = new2["sentence2"]

new_test = pd.merge(sentences2, text_new22, left_index=True, right_index=True)


new_test.to_csv("test_3vtfidfv1.csv", index=False)


#  Test code, used to validate results but not written to CSV and not
#  used in any follow-up downstream analysis


extra_weight = [
    "tire",
    "save",
    "fuel",
    "rear",
    "front",
    "tight",
    "loose",
    "loos",
    "adjust",
    "adjustments",
    "stage",
    "bouncing",
    "bounc",
    "321",
    "pit",
]

text_weights = []
for i, post in enumerate(text2):
    # post = punc.sub( '', post ).lower()    # Remove punctuation, convert to lowercase
    # print(post)
    #  Replace original news string with no punctuation, lowercase, stop word removed, stemmed version
    t_vec = 0
    number_of_terms = 0
    for term in nltk.word_tokenize(post):
        if term in names:
            t_vec = t_vec + list(df_idf.loc[term])[0]
        # if term in extra_weight:
        # t_vec = t_vec + 5
        if term in names:
            number_of_terms = number_of_terms + 1
            # if term in extra_weight:
            # number_of_terms = number_of_terms + 1
    if number_of_terms != 0:
        if number_of_terms == 1:
            t_vec = 0
        elif number_of_terms <= 3:
            t_vec = t_vec / (number_of_terms)
        else:
            t_vec = t_vec / number_of_terms

    # print(t_vec)
    # s[0][ i ] = ' '.join( t_vec )
    text_weights.append(t_vec)


# extra_weight2 = ['save fuel', '321', 'pit']

for i in range(len(text2)):
    for j in extra_weight:
        if j in text2[i]:
            print(text2[i])
            text_weights[i] = text_weights[i] + 10

sentences = pd.DataFrame({"sentence": text2, "weight": text_weights})

y = sentences.sort_values(by=["weight"])


summary = sentences.loc[(sentences["weight"] > 1.5)]

text_new2 = pd.DataFrame({"sentence2": text_all})

new = pd.merge(summary, text_new2, left_index=True, right_index=True)

xxx = new["sentence2"]
