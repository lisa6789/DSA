# lda vs nmf
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import glob
from tika import parser
import os
from nltk.corpus import stopwords
import re

doclist = []
input_path = 'C:\\test'
stop_words = set(stopwords.words('english'))


# Use Tika to parse the file
def parsewithtika(inputfile):
    parsed = parser.from_file(inputfile)
    # Extract the text content from the parsed file
    psd = parsed["content"]
    return re.sub(r'\s+', ' ', psd)


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


print('Starting processing - the following files have been processed:')
for input_file in glob.glob(os.path.join(input_path, '*.*')):
    # Grab the file name
    filename = os.path.basename(input_file)
    fname = os.path.splitext(filename)[0]
    print(filename)

    # Parse the file to get to the text
    parsed = parsewithtika(input_file)
    doclist.append(parsed)


documents = doclist
print(documents[0])

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)
