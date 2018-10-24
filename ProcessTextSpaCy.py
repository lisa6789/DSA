import glob
from tika import parser
import os
import nltk
from nltk import sent_tokenize
import re

import spacy

nlp = spacy.load('en_core_web_sm')


input_path = 'C:\\t3'


# Use Tika to parse the file
def parsewithtika(inputfile):
    parsed = parser.from_file(inputfile)
    # Extract the text content from the parsed file
    psd = parsed["content"]
    return re.sub(r'\s+', ' ', psd)


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



    sent_text = nltk.sent_tokenize(parsed)  # this gives us a list of sentences
    # now loop over each sentence and tokenize it separately
    for sentence in sent_text:
        if "IS" in sentence:
            processed_spacy = nlp(sentence)
            for token in processed_spacy:
                print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                    token.shape_, token.is_alpha, token.is_stop)






