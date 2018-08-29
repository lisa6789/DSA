import glob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from tika import parser
import nltk
import re
import os
from langdetect import detect


input_path = 'C:\\test'


def parsewithtika(inputfile):
    parsed = parser.from_file(inputfile)
    # Extract the text content from the parsed file
    psd = parsed["content"]
    return re.sub(r'\s+', ' ', psd)


# Language filter
def filterlanguage(inputfile):
    if detect(inputfile) != 'en':
        return True
    return False


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

    stemmer = SnowballStemmer("english")
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(parsed)

    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue

        word = stemmer.stem(word)

        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    sentences = sent_tokenize(parsed)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    # Average value of a sentence from original text
    average = int(sumValues / len(sentenceValue))

    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence

    print(summary)