import glob
from tika import parser
import os
import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
from langdetect import detect
import pandas as pd
import string


# document tokenization after text pre-preprocessing to differentiate types then token based on type

input_path = 'C:\\test'
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

d = list()


# Use Tika to parse the file
def parsewithtika(inputfile):
    parsed = parser.from_file(inputfile)
    # Extract the text content from the parsed file
    psd = parsed["content"]
    # Convert double newlines into single newlines
    psd.replace('\n\n', '\n')
    return psd


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
    dataframe['words'] = dataframe['sentences'].apply(word_tokenize)
    dataframe['words'] = dataframe['words'].apply(lambda x: [item.lower() for item in x])
    dataframe['words'] = dataframe['words'].apply(lambda x: [item.strip(string.punctuation) for item in x])
    dataframe['words'] = dataframe['words'].apply(lambda x: [item for item in x if item.isalpha()])
    dataframe['words'] = dataframe['words'].apply(lambda x: [item for item in x if item not in stop_words])
    dataframe['pos'] = dataframe['words'].apply(nltk.pos_tag)
    return dataframe


# Main loop function
# Iterate over all files in the folder and process each one in turn
for input_file in glob.glob(os.path.join(input_path, '*.*')):
    # Grab the file name
    filename = os.path.basename(input_file)
    fname = os.path.splitext(filename)[0]
    print(filename)
    print(fname)

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
    d.append(pd.DataFrame({'document': filename, 'sentences': sentences}))

    # check for output folder and build if it doesn't exist
    # if not os.path.exists(input_path + '\\output\\' + fname):
    #     os.makedirs(input_path + '\\output\\' + fname)
    # # write out the text extracted by tika
    # f = open(input_path + '\\output\\' + '\\' + fname + '\\' + os.path.splitext(filename)[0] + '.txt', 'wb')
    # f.write(parsed.encode('utf-8').strip())
    # f.close()

# Create full dataframe
doc = pd.concat(d)

# Word tokenize the sentences, cleanup, parts of speech tagging
wordtokens(doc)

print(doc.head())



