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
from sklearn.cluster import KMeans
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.corpus import brown
from nltk.tag import RegexpTagger
from nltk.tag import UnigramTagger
from nltk.stem import PorterStemmer

regexp_tagger = RegexpTagger(
            [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
             (r'(The|the|A|a|An|an)$', 'AT'),   # articles
             (r'.*able$', 'JJ'),                # adjectives
             (r'.*ness$', 'NN'),                # nouns formed from adjectives
             (r'.*ly$', 'RB'),                  # adverbs
             (r'.*s$', 'NNS'),                  # plural nouns
             (r'.*ing$', 'VBG'),                # gerunds
             (r'.*ed$', 'VBD'),                 # past tense verbs
             (r'.*', 'NN')                      # nouns (default)
        ])
brown_train = brown.tagged_sents(categories='news')
unigram_tagger = UnigramTagger(brown_train, backoff=regexp_tagger)


# Define location of files and keywords - TODO parameterise these
stemmer = SnowballStemmer('english')
pstemmer = PorterStemmer()

input_path = 'C:\\test'
stop_words = set(stopwords.words('english'))
keywords = ['IS', 'terrorism', 'bomb', 'is', 'the', 'consortium']
poskeywords = unigram_tagger.tag(keywords)
stemkeywords = unigram_tagger.tag([pstemmer.stem(t) for t in keywords])


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
    dataframe['allwords'] = dataframe['words'].apply(lambda x: [item.strip(string.punctuation).lower()
                                                                for sublist in x for item in sublist])
    dataframe['allwords'] = (dataframe['allwords'].apply(lambda x: [item for item in x if item.isalpha()
                                                                    and item not in stop_words]))
    dataframe['mfreq'] = dataframe['allwords'].apply(nltk.FreqDist)
    dataframe['poslist'] = dataframe['pos'].apply(lambda x: [item for sublist in x for item in sublist])
    dataframe['mfreqpos'] = dataframe['poslist'].apply(nltk.FreqDist)
    dataframe['stemwords'] = dataframe['words'].apply(lambda x: [pstemmer.stem(item) for sublist in x
                                                                 for item in sublist])
    dataframe['stemwords'] = (dataframe['stemwords'].apply(lambda x: [item for item in x if item.isalpha()
                                                                      and item not in stop_words]))
    dataframe['mfreqstem'] = dataframe['stemwords'].apply(nltk.FreqDist)

    return dataframe


# Score documents based on cleansed dataset - so should discount stopwords and be sensible
def scoring(dataframe):
    word_matches = defaultdict(list)
    for word in keywords:
        for idx, row in dataframe.iterrows():
            if word in row['allwords']:
                dataframe.loc[idx, 'score'] += (row['mfreq'][word] * 0.75)
                if not row['document'] in word_matches[word]:
                    word_matches[word].append(row['document'])
    print('\n')
    print('The following keyword hits occurred:')

    for key, val in word_matches.items():
        print("Keyword: " + key + ". Found in these documents: ")
        print(val)

    return dataframe


# Score documents based on pos - should be most exact match
def scoringpos(dataframe):
    word_matches = defaultdict(list)
    for (w1, t1) in poskeywords:
        for idx, row in dataframe.iterrows():
            if (w1, t1) in row['poslist']:
                dataframe.loc[idx, 'score'] += row['mfreqpos'][(w1, t1)]
                if not row['document'] in word_matches[w1]:
                    word_matches[w1].append(row['document'])
    print('\n')
    print('The following keyword hits occurred:')

    for key, val in word_matches.items():
        print("Keyword: " + key + ". Found in these documents: ")
        print(val)

    return dataframe


# Score documents based on cleansed dataset - so should discount stopwords and be sensible
def scoringstem(dataframe):
    word_matches = defaultdict(list)
    for word in keywords:
        for idx, row in dataframe.iterrows():
            if word in row['stemwords']:
                dataframe.loc[idx, 'score'] += (row['mfreqstem'][word] * 0.5)
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


def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# Cluster documents and demonstrate prediction
# TODO - calculate ideal k value
def clustering(documents):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=0.2, use_idf=True,
                                 tokenizer=tokenize_and_stem, ngram_range=(1, 3))
    X = vectorizer.fit_transform(doclist)

    true_k = 5
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

    Y = vectorizer.transform(["this is a document about islamic state "
                              "and terrorists and bombs IS jihad terrorism isil"])
    prediction = model.predict(Y)
    print("A document with 'bad' terms would be in:")
    print(prediction)

    Y = vectorizer.transform(["completely innocent text just about kittens and puppies"])
    prediction = model.predict(Y)
    print("A document with 'good' terms would be in:")
    print(prediction)


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


def nmflda(documentlist):
    no_features = 1000

    # NMF is able to use tf-idf
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documentlist)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(documentlist)
    tf_feature_names = tf_vectorizer.get_feature_names()

    no_topics = 5

    # Run NMF
    nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

    # Run LDA
    lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5,
                                    learning_method='online', learning_offset=50.,random_state=0).fit(tf)

    no_top_words = 10
    print("NMF Topics: ")
    display_topics(nmf, tfidf_feature_names, no_top_words)
    print("LDA Topics: ")
    display_topics(lda, tf_feature_names, no_top_words)


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

    # Language detection algorithm is non - deterministic, which means that if you try to run it on a text which is
    # either too short or too ambiguous, you might get different results every time you run it
    if filterlanguage(parsed):
        continue

    tokenised = tokenmakerwords(parsed)

    # Ignore any documents with <50 words
    if len(tokenised) < 100:
        continue

    # Create doclist for use in topic modelling
    doclist.append(parsed)
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

# Now we score in a calculated manner:
# TODO - separate out the word matches printing
# Score 1 for matching word (case sensitive and POS)
scoringpos(d)
# Score 0.75 for matching word (case insensitive,  stop words removed)
scoring(d)
# Score 0.5 for matching stem of word (case insensitive, stop words removed
scoringstem(d)

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

# Print results of K Means Cluster and prediction modelling
clustering(doclist)

# Print results of NMF vs LDA topic modelling
nmflda(doclist)

