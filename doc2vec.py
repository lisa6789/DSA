from os import listdir
import gensim
from tika import parser
import os

LabeledSentence = gensim.models.doc2vec.LabeledSentence

input_path = 'C:\\test'

def parsewithtika(inputfile):
    parsed = parser.from_file(inputfile)
    # Extract the text content from the parsed file
    psd = parsed["content"]
    return psd

docLabels = []
docLabels = [f for f in listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]



data = []
for doc in docLabels:
    data.append(parsewithtika(input_path + '\\' + doc))


class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(words=doc.split(),tags=[self.labels_list[idx]])


it = LabeledLineSentence(data, docLabels)

model = gensim.models.Doc2Vec(size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025)
# use fixed learning rate
model.build_vocab(it)
for epoch in range(10):
    model.train(it, total_examples=model.corpus_count, epochs=model.iter)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no deca
    model.train(it, total_examples=model.corpus_count, epochs=model.iter)

model.save('doc2vec.model')

model['START_TranscendingOrganizationIndividualsandtheIslamicState_AnalyticalBrief_June2014.pdf']
