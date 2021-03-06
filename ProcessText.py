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
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import en_core_web_sm  # or any other model you downloaded via spacy download or pip
from fpdf import FPDF

# Set up PDF file - requires the DejaVu font to be installed in a fonts folder in the
# fpdf package directory in the python environment
pdf = FPDF()
pdf.add_page()
pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
pdf.add_font('DejaVuSans-Bold', '', 'DejaVuSans-Bold.ttf', uni=True)
pdf.set_font('DejaVu', '', 14)
pdf.cell(w=0, txt="Output Report", ln=1, align="C")
pdf.ln(20)

# Load the model to be used by SpaCy for Named Entity Recognition
nlp = en_core_web_sm.load()

# Set up a stemmer to use to stem words
pstemmer = PorterStemmer()

# Set up input path, stop words to be excluded and keywords to be matched
input_path = 'C:\\t2'
stop_words = set(stopwords.words('english'))
keywords = ['IS', 'terrorism', 'bomb', 'is', 'the', 'consortium']

# Filter out stopwords from keywords list, POS tag keywords
filterkeywords = [w for w in keywords if w not in stop_words]
poskeywords = nltk.pos_tag(filterkeywords)

# If the first keyword is a verb, move it and reparse the list
# This prevents verbs that may also be nouns being misidentified
if poskeywords[0][1] == 'VBZ':
    filterkeywords.insert(1, filterkeywords.pop(0))
    poskeywords = nltk.pos_tag(filterkeywords)

# Build a list of stem keywords for matching
stemkeywords = nltk.pos_tag([pstemmer.stem(t) for t in filterkeywords])

# Set up Dataframe - this will hold all the documents and the scores
d = pd.DataFrame()

# Create a list to use for clustering - this is for topic modelling
doclist = []
word_matches = defaultdict(list)


# Use Tika to parse the file
def parsewithtika(inputfile):
    parsed = parser.from_file(inputfile)
    # Extract the text content from the parsed file
    psd = parsed["content"]
    return re.sub(r'\s+', ' ', psd)


# Language filter - removes non english documents from the list
def filterlanguage(inputfile):
    if detect(inputfile) != 'en':
        return True
    return False


# Get parts of speech from SpaCy
def pos(x):
    return [(token.text, token.tag_) for token in x]


# Add the parts of speech to the words
def spacy_pos(x):
    pos_sent = []
    for sentence in x:
        processed_spacy = nlp(sentence)
        pos_sent.append(pos(processed_spacy))
    return pos_sent


# Add NER tags to words, and return the set so we don't have duplicates
def ner(x):
    ents = []
    for sentence in x:
        processed_spacy = nlp(sentence)
        for ent in processed_spacy.ents:
            ents.append((ent.text, ent.label_))
    return set(ents)


# Word tokens, parts of speech tagging
def wordtokens(dataframe):
    # Get all the words
    dataframe['words'] = (dataframe['sentences'].apply(lambda x: [word_tokenize(item) for item in x]))
    # Get all the parts of speech tags
    dataframe['pos'] = dataframe['sentences'].map(spacy_pos)
    # Get all the named entity tags
    dataframe['ner'] = dataframe['sentences'].map(ner)
    # Lowercase every word and put them all in a single list for each document
    dataframe['allwords'] = dataframe['words'].apply(lambda x: [item.strip(string.punctuation).lower()
                                                                for sublist in x for item in sublist])
    # Strip out non words and stop words
    dataframe['allwords'] = (dataframe['allwords'].apply(lambda x: [item for item in x if item.isalpha()
                                                                    and item not in stop_words]))
    # Calculate the frequency of each word in the document
    dataframe['mfreq'] = dataframe['allwords'].apply(nltk.FreqDist)
    # Get all the pos tagged words in a single list for each document
    dataframe['poslist'] = dataframe['pos'].apply(lambda x: [item for sublist in x for item in sublist])
    # Calculate the frequency of each pos tagged word
    dataframe['mfreqpos'] = dataframe['poslist'].apply(nltk.FreqDist)
    # Get the stems of all the words
    dataframe['stemwords'] = dataframe['words'].apply(lambda x: [pstemmer.stem(item) for sublist in x
                                                                 for item in sublist])
    # Get rid of non words and stop words
    dataframe['stemwords'] = (dataframe['stemwords'].apply(lambda x: [item for item in x if item.isalpha()
                                                                      and item not in stop_words]))
    # Calculate frequency of stemmed words
    dataframe['mfreqstem'] = dataframe['stemwords'].apply(nltk.FreqDist)

    return dataframe


# Score documents based on cleansed dataset - so should discount stopwords and be sensible
def scoring(dataframe, list):
    for word in keywords:
        for idx, row in dataframe.iterrows():
            if word in row['allwords']:
                if not row['document'] in list[word]:
                    list[word].append(row['document'])
                    dataframe.loc[idx, 'score'] += (row['mfreq'][word] * 0.75)
    return dataframe


# Score documents based on pos - should be most exact match
def scoringpos(dataframe, list):
    for (w1, t1) in poskeywords:
        for idx, row in dataframe.iterrows():
            if (w1, t1) in row['poslist']:
                if not row['document'] in list[w1]:
                    list[w1].append(row['document'])
                    dataframe.loc[idx, 'score'] += row['mfreqpos'][(w1, t1)]
    return dataframe


# Score documents based on stemmed words in cleansed dataset - so should discount stopwords and be sensible
def scoringstem(dataframe, list):
    for word in stemkeywords:
        for idx, row in dataframe.iterrows():
            if word in row['stemwords']:
                if not row['document'] in list[word]:
                    list[word].append(row['document'])
                    dataframe.loc[idx, 'score'] += (row['mfreqstem'][word] * 0.5)
    return dataframe


# Find keywords using POS, show the sentence the word was found in
def contextkeywords(dataframe):
    pdf.set_font('DejaVuSans-Bold', '', 14)
    pdf.cell(w=0,txt="Here are the exact keyword matches in context: ", ln=1, align="L")
    pdf.ln(10)
    for (w1, t1) in poskeywords:
        for idx, row in dataframe.iterrows():
            for index, r in enumerate(row['pos']):
                if (w1, t1) in r:
                    pdf.set_font('DejaVuSans-Bold', '', 12)
                    pdf.multi_cell(w=0, h=10, txt=row['document'] + ' - ', align="L")
                    pdf.set_font('DejaVu', '', 12)
                    pdf.multi_cell(w=0, h=10, txt=' '.join(row['words'][index]),  align="L")
                    pdf.ln(5)
    return dataframe


# Show all the documents that had keyword matches, for each keyword
def printkeywordmatches(list):
    for key, val in list.items():
        print("Keyword: " + key + ". Found in these documents: ")
        pdf.multi_cell(w=0, h=10, txt="Keyword: " + key + ". Found in these documents: ", align="L")
        pdf.ln(5)
        print(val)
        pdf.multi_cell(w=0, h=10, txt=', '.join(val), align="L")
        pdf.ln(10)


# tokenize each word in the text and then filter out non alphabet words, then get all the stems
def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [pstemmer.stem(t) for t in filtered_tokens]
    return stems


# Cluster documents and demonstrate prediction
# K means clustering using the following parameters:
# - filter out english stopwords
# - word must appear in no more than 80% of documents
# - word must appear in no less than 20% of documents
# - token and stem words, allow up to 3 words together as a grouping
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

    # Ignore any documents with <100 words
    if len(parsed) < 100:
        continue

    # Create doclist for use in topic modelling
    doclist.append(parsed)
    # Sentence fragments
    sentences = sent_tokenize(parsed)

    # Build up dataframe
    temp = pd.Series([filename, sentences])
    d = d.append(temp, ignore_index=True)

print('\n')
d.reset_index(drop=True, inplace=True)
d.columns = ['document', 'sentences']


# Word tokenize the sentences, cleanup, parts of speech tagging
wordtokens(d)
d['score'] = 0

# Now we score in a calculated manner:
# Score 1 for matching word (case sensitive and POS)
scoringpos(d, word_matches)
# Score 0.75 for matching word (case insensitive,  stop words removed)
scoring(d, word_matches)
# Score 0.5 for matching stem of word (case insensitive, stop words removed)
scoringstem(d, word_matches)
# Print out the results of keyword matching
printkeywordmatches(word_matches)
# Find words in context with POS
contextkeywords(d)

# Sort by scoring
d = d.sort_values('score', ascending=False)

# Print sorted documents
print('\n')
pdf.ln(10)
print('Here are the scores based on cleansed data:')
pdf.multi_cell(w=0, h=10, txt='Here are the scores based on cleansed data:', align="L")
pdf.ln(5)
print(d[['document', 'score']])
pdf.multi_cell(w=0, h=10, txt=d[['document', 'score']].to_string(), align="L")
pdf.ln(10)

# cater for small no of docs
# cater for 0 scores
topdocs = d.head(int(len(d)*0.1))

# Print results of NER for people
print('People discovered:')
pdf.multi_cell(w=0, h=10, txt='People discovered:', align="L")
pdf.ln(5)
for doc in topdocs['ner']:
    for (a,b) in doc:
        if b == 'PERSON':
            print(a)
            pdf.multi_cell(w=0, h=10, txt=a, align="L")
pdf.ln(10)

# Print results of NER for organisations
print('Orgs discovered:')
pdf.multi_cell(w=0, h=0, txt='Orgs discovered:', align="L")
pdf.ln(5)
for doc in topdocs['ner']:
    for (a,b) in doc:
        if b == 'ORG':
            print(a)
            pdf.multi_cell(w=0, h=10, txt=a, align="L")

# Output the case document with all the printed results to PDF
pdf.output('C:\\tout\\simple_demo.pdf')