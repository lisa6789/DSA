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
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Define location of files and keywords - TODO parameterise these
input_path = 'C:\\test'
stop_words = set(stopwords.words('english'))
keywords = ['IS', 'terrorism', 'bomb', 'is', 'the', 'consortium']

# Set up Dataframe
d = pd.DataFrame()

# Create a list to use for clustering
doclist = []


# Use Tika to parse the file
def parsewithtika(inputfile):
    parsed = parser.from_file(inputfile)
    # Extract the text content from the parsed file
    psd = parsed["content"]
    return re.sub(r'\s+', ' ', psd)


# Return NLTK text from the document - used to filter out short documents but may
# also be used for further processing in future dev
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
    dataframe['words'] = (dataframe['sentences'].apply(lambda x: [word_tokenize(item) for item in x]))
    dataframe['pos'] = dataframe['words'].apply(lambda x: [nltk.pos_tag(item) for item in x])
    dataframe['allwords'] = d['words'].apply(lambda x: [item.strip(string.punctuation).lower() for sublist in x for item in sublist])
    dataframe['allwords'] = (dataframe['allwords'].apply(lambda x: [item for item in x if item.isalpha()
                                                               and item not in stop_words]))
    dataframe['mfreq'] = d['allwords'].apply(nltk.FreqDist)
    return dataframe


# Score documents based on cleansed dataset - so should discount stopwords and be sensible
def scoring(dataframe):
    word_matches = defaultdict(list)
    for word in keywords:
        for idx, row in dataframe.iterrows():
            if word in row['allwords']:
                dataframe.loc[idx, 'score'] += row['mfreq'][word]
                if not row['document'] in word_matches[word]:
                    word_matches[word].append(row['document'])
    print('\n')
    print('The following keyword hits occurred:')

    for key, val in word_matches.items():
        print("Keyword: " + key + ". Found in these documents: ")
        print(val)

    return dataframe


# Find keywords using POS
def contextkeywords(dataframe):
    print('\n')
    print('Here are the keywords in context: ')
    # Search for IS as a noun
    for idx, row in dataframe.iterrows():
        for index, r in enumerate(row['pos']):
            for (w1, t1) in r:
                if w1 == 'IS' and t1 == 'NNP':
                    print(row['document'] + ' - ' + ' '.join(row['words'][index]))
                    print('\n')

    return dataframe


# Sort using a dirty model
def dirtyscoring(dataframe):
    dataframe['score2'] = 0
    dataframe['w2'] = dataframe['words'].apply(lambda x: [item for sublist in x for item in sublist])
    dataframe['mfreq2'] = dataframe['w2'].apply(nltk.FreqDist)

    word_matches = defaultdict(list)
    for word in keywords:
        for idx, row in dataframe.iterrows():
            if word in row['w2']:
                dataframe.loc[idx, 'score2'] += row['mfreq2'][word]
                if not row['document'] in word_matches[word]:
                    word_matches[word].append(row['document'])
    print('\n')
    print('The following keyword hits occurred in the uncleansed data:')

    for key, val in word_matches.items():
        print("Keyword: " + key + ". Found in these documents: ")
        print(val)

    return dataframe


# Cluster documents and demonstrate prediction
# TODO - calculate ideal k value
def clustering(documents):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(doclist)

    true_k = 3
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]),
        print

    print("\n")
    print("Prediction")

    Y = vectorizer.transform(["this is a document about islamic state and terrorists and bombs IS jihad terrorism isil"])
    prediction = model.predict(Y)
    print(prediction)

    Y = vectorizer.transform(["completely innocent text just about kittens and puppies"])
    prediction = model.predict(Y)
    print(prediction)


# Main loop function
# Iterate over all files in the folder and process each one in turn
print('Starting processing - the following files have been processed:')
for input_file in glob.glob(os.path.join(input_path, '*.*')):
    # Grab the file name
    filename = os.path.basename(input_file)
    fname = os.path.splitext(filename)[0]
    print(filename)

    # Parse the file to get to the text
    parsed = parsewithtika(input_file)
    doclist.append(parsed)

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
d['score'] = 0

# Add scoring
# TODO - use POS/stemming to make better counts of words, deal with cases
scoring(d)

# Find words in context with POS
contextkeywords(d)

# Sort by scoring
d = d.sort_values('score', ascending=False)

# Print sorted documents
print('\n')
print('Here are the scores based on cleansed data:')
print(d[['document', 'score']])

dirtyscoring(d)

d = d.sort_values('score2', ascending=False)
print('\n')
print('Here are the scores based on uncleansed data:')
print(d[['document', 'score2']])

clustering(doclist)




