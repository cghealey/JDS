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

    if new_data[i]["type"] == "pronunciation" and i == 0:
        subcluster.append(new_data[i]["alternatives"][0]["content"])
        start_time.append(new_data[i]["start_time"])
    elif new_data[i]["type"] == "pronunciation" and i != 0:
        if (
            new_data[i - 1]["type"] == "pronunciation"
            and float(new_data[i - 1]["end_time"])
            >= float(new_data[i]["start_time"]) - 5
        ):
            subcluster.append(new_data[i]["alternatives"][0]["content"])
        elif new_data[i - 1]["type"] == "punctuation":
            if (
                float(new_data[i - 2]["end_time"])
                >= float(new_data[i]["start_time"]) - 5
            ):
                subcluster.append(new_data[i]["alternatives"][0]["content"])
            else:
                data1[j] = " ".join(subcluster)
                subcluster = []
                j += 1
                subcluster.append(new_data[i]["alternatives"][0]["content"])
                start_time.append(new_data[i]["start_time"])
        else:
            data1[j] = " ".join(subcluster)
            subcluster = []
            j += 1
            subcluster.append(new_data[i]["alternatives"][0]["content"])
            start_time.append(new_data[i]["start_time"])
    elif new_data[i]["type"] == "punctuation":
        subcluster.append(new_data[i]["alternatives"][0]["content"])

data1[-1] = " ".join(subcluster)


data1 = [i for i in data1 if i != 0]

for i in range(len(data1)):
    data1[i] = re.sub(r'\s+([?.!",])', r"\1", data1[i])


# text2 =  text.split(".")

text2 = data1

for i in range(len(text2)):
    if "green" in text2[i] or "Green" in text2[i]:
        text2 = text2[i:]
        start_time = start_time[i:]
        break

##############################################################################

from nltk import tokenize

new_start_time = []
for i in range(len(text2)):
    text2[i] = tokenize.sent_tokenize(text2[i])
    new_start_time += [start_time[i]] * len(text2[i])

text2 = [val for sublist in text2 for val in sublist]
start_time = new_start_time.copy()

##############################################################################


#########################################################################

duplicates = []
for i in range(len(text2)):
    if i > 0:
        if text2[i - 1] == text2[i]:
            duplicates.append(i)

duplicates.reverse()

for i in duplicates:
    start_time.pop(i)
    text2.pop(i)


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


punc = re.compile("[%s]" % re.escape(string.punctuation))
stop_w = nltk.corpus.stopwords.words("english")
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

text3 = [x for x in text2 if x != ""]
text_all = [x for x in text_all if x != ""]
text4 = [x for x in text3 if len(x) > 2]
text_all = [x for x in text_all if len(x) > 2]

new_start_time = []
new_text_all = []
for i, x in enumerate(text_all):
    if len(x) <= 2:
        continue
    new_text_all += [x]
    new_start_time += [start_time[i]]

text_all = new_text_all.copy()
start_time = new_start_time.copy()
#################################################

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

text_weights = [0] * len(text2)
for i in range(len(text2)):
    for j in key_words:
        if j in text2[i]:
            # print(text2[i])
            text_weights[i] = text_weights[i] + 10


sentences = pd.DataFrame({"sentence": text2, "weight": text_weights})
sentences = pd.DataFrame(
    {"sentence": text2, "weight": text_weights, "time": start_time}
)

# y = sentences.sort_values(by=['weight'])

summary = sentences.loc[(sentences["weight"] > 0)]

text_new2 = pd.DataFrame({"sentence2": text_all})

new = pd.merge(summary, text_new2, left_index=True, right_index=True)

text_weights = [0] * len(text2)
for i, txt in enumerate(text2):
    for j in key_words:
        if j in txt:
            # print(text2[i])
            text_weights[i] = text_weights[i] + 10

xxx = new["sentence2"]

##############################################################################
sentences = pd.DataFrame(
    {"sentence": text_all, "weight": text_weights, "time": start_time}
    #    {"sentence": new["sentence"], "weight": text_weights, "time": new["time"]}
)
sentences.to_csv("test_3v1.csv", index=False)


##############################################################################

for i in range(len(text2)):
    if text_weights[i] == 0:
        text2[i] = ""

##############################################################################
##############################################################################
##############################################################################

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

for i, post in enumerate(text2):
    post = punc.sub(
        "", post
    ).lower()  # Remove punctuation, convert to lowercase
    # print(post)
    #  Replace original news string with no punctuation, lowercase, stop word removed, stemmed version
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
    if len(t_vec) > 5:
        text2[i] = " ".join(t_vec)
    else:
        text2[i] = " "
    # text_all[ i ] = ' '.join(t_vec2)

empty = []
for i in range(len(text2)):
    if text2[i] == " ":
        empty.append(i)

empty.reverse()

for i in empty:
    text2.pop(i)
    text_all.pop(i)
    start_time.pop(i)


# text2 = [x for x in text2 if x !=' ']
# text_all = text2.copy()

vectorizer = CountVectorizer()

X_tf = vectorizer.fit_transform(text2)
shp = X_tf.toarray().shape
print(f"Documents: {shp[0]}; Unique Terms: {shp[1]}")

terms = vectorizer.get_feature_names_out()
terms = list(terms)

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


x_norm = []

length = np.linalg.norm(tfidfMatrix, axis=1)  #  List of document vector lengths

for i in range(
    0, tfidfMatrix.shape[0]
):  #  For each document, normalize its vector
    x_norm.append([x / length[i] for x in tfidfMatrix[i]])

###########################Different TFIDF######################################

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

x_norm = []

length = np.linalg.norm(X_tfidf, axis=1)  #  List of document vector lengths

for i in range(0, X_tfidf.shape[0]):  #  For each document, normalize its vector
    x_norm.append([x / length[i] for x in X_tfidf[i]])

##############################################################################
##############################################################################
##############################################################################

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


sentences2 = pd.DataFrame({"sentence": text2, "weight": x_weight})

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


# print idf values
df_idf = pd.DataFrame(
    tfidf.idf_, index=vectorizer.get_feature_names_out(), columns=["idf_weights"]
)

# sort ascending
df_idf.sort_values(by=["idf_weights"])

names = vectorizer.get_feature_names_out()

sentence_weights = []
for i, post in enumerate(text2):
    # post = punc.sub( '', post ).lower()    # Remove punctuation, convert to lowercase
    # print(post)
    #  Replace original news string with no punctuation, lowercase, stop word removed, stemmed version
    t_vec = 0
    number_of_terms = 0
    for term in nltk.word_tokenize(post):
        if term in names:
            t_vec = t_vec + list(df_idf.loc[term])[0]
        if term in key_words:
            t_vec = t_vec + 5
        if term in names:
            number_of_terms = number_of_terms + 1
            # if term in extra_weight:
            # number_of_terms = number_of_terms + 1
    if number_of_terms != 0:
        # if number_of_terms == 1:
        # t_vec = 0
        # elif number_of_terms <= 3:
        # t_vec = t_vec/(number_of_terms)
        # else:
        t_vec = t_vec / number_of_terms

    # print(t_vec)
    # s[0][ i ] = ' '.join( t_vec )
    sentence_weights.append(t_vec)

sentences2 = pd.DataFrame({"sentence": text2, "weight": sentence_weights})

y2 = sentences2.sort_values(by=["weight"])

summary2 = sentences2.loc[(sentences2["weight"] >= 5)]

text_new22 = pd.DataFrame({"sentence2": text_all})

new2 = pd.merge(summary2, text_new22, left_index=True, right_index=True)

xxx2 = new2["sentence2"]

new_test = pd.merge(sentences2, text_new22, left_index=True, right_index=True)


new_test.to_csv("test_3vtfidfv1.csv", index=False)


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
