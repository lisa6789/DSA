import glob
from tika import parser
import os
import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
from langdetect import detect
import pandas as pd
import string
import re
from nltk.corpus import stopwords

# document tokenization after text pre-preprocessing to differentiate types then token based on type

input_path = 'C:\\test'
stop_words = set(stopwords.words('english'))

# have df be document, sentences, words, pos
# do keyword searching from list
# contextualise search using pos
d = pd.DataFrame()

# Use Tika to parse the file
def parsewithtika(inputfile):
    parsed = parser.from_file(inputfile)
    # Extract the text content from the parsed file
    psd = parsed["content"]
    return re.sub(r'\s+', ' ', psd)


def tokenmakerwords(inputfile):
    # Create tokens
    tokens = word_tokenize(inputfile)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    import string
    stripped = [w.strip(string.punctuation) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    words = [w for w in words if w not in stop_words]
    text = nltk.Text(words)
    return text


# Language filter
def filterlanguage(inputfile):
    if detect(inputfile) != 'en':
        return True
    return False


# Word tokens, parts of speech tagging
def wordtokens(dataframe):
    dataframe['words'] = (dataframe['sentences'].apply(lambda x: [word_tokenize(item.strip(string.punctuation).lower())
                                                                  for item in x]))
    dataframe['words'] = (dataframe['words'].apply(lambda x: [[item for item in lst if item.isalpha()
                                                               and item not in stop_words] for lst in x]))
    dataframe['pos'] = dataframe['words'].apply(lambda x: [nltk.pos_tag(item) for item in x])
    dataframe['allwords'] = d['words'].apply(lambda x: [item for sublist in x for item in sublist])
    dataframe['mfreq'] = d['allwords'].apply(nltk.FreqDist)
    return dataframe


# Main loop function
# Iterate over all files in the folder and process each one in turn
for input_file in glob.glob(os.path.join(input_path, '*.*')):
    # Grab the file name
    filename = os.path.basename(input_file)
    fname = os.path.splitext(filename)[0]
    print(filename)

    # Parse the file to get to the text
    parsed = parsewithtika(input_file)

    # Language detection algorithm is non - deterministic, which means that if you try to run it on a text which is
    # either too short or too ambiguous, you might get different results every time you run it
    if filterlanguage(parsed):
        continue

    tokenised = tokenmakerwords(parsed)

    # Ignore any documents with <50 words
    if len(tokenised) < 100:
        continue

    # Sentence fragments
    sentences = sent_tokenize(parsed)

    # Build up dataframe
    temp = pd.Series([filename, sentences])
    d = d.append(temp, ignore_index=True)

d.reset_index(drop=True, inplace=True)
d.columns = ['document', 'sentences']

# Word tokenize the sentences, cleanup, parts of speech tagging
wordtokens(d)

print(d.head())

keywords = ['IS', 'terrorism', 'bomb', 'is', 'the']

from collections import defaultdict
word_matches = defaultdict(list)
for word in keywords:
    for idx, row in d.iterrows():
        for wordlist in row['words']:
            if word in wordlist and not row['document'] in word_matches[word]:
                word_matches[word].append(row['document'])

for key, val in word_matches.items():
    print(key, val)



