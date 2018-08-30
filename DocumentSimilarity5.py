import os, json, numpy
import nltk
import enchant
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import gensim
from scipy import spatial
import string

'''
This model calculates average word vector similarity between documents.
Top 5 nouns from Tika generated abstracts are considered. Average
word vectors are built from those words which are representative of
a document. Pairwise similarity is calculated between each pair.
'''

def norm(v):
  sum = float(0)
  for i in range(len(v)):
    sum += v[i]**2
  return sum**(0.5)

def dot(v1, v2):
    return sum(x*y for x,y in zip(v1,v2))

def avg_feature_vector(tokens, model, num_features, index2word_set):
    words = [w.lower() for w in tokens if not w.lower() in stopwords.words('english')]
    feature_vec = numpy.zeros((num_features,), dtype='float32')
    #print words
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = numpy.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = numpy.divide(feature_vec, n_words)
    return feature_vec

def getNouns(abstract):
    d = enchant.Dict("en_US")
    sentences = nltk.sent_tokenize(abstract.decode('utf-8'))  # tokenize sentences
    nouns = set()  # empty to array to hold all nouns
    frequency_nouns = {}
    for sentence in sentences:
        for word, pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                if d.check(word):
                    if word not in nouns:
                        nouns.add(word)
    for _word in nouns:
        if _word in frequency_nouns:
            frequency_nouns[_word] += 1
        else:
            frequency_nouns[_word] = 1
    sorted_keys = sorted(frequency_nouns, key=frequency_nouns.get, reverse=True)
    return list(nouns),sorted_keys


if __name__ == '__main__':
    printable = set(string.printable)
    path_to_json = '/Users/siddhartharoynandi/Desktop/RA_2018/new_data_json/'
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    print json_files
    file_nouns = []
    for file in json_files:
        data = json.load(open(path_to_json + str(file)))
        content_data = filter(lambda x: x in printable, data[0]['X-TIKA:content'])
        content_nouns, content_nouns_freq = getNouns(content_data)
        file_nouns.append(content_nouns_freq[:5])
    #print file_nouns

    ################# Average Word2Vec Section ########################
    content_vectors = []
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    index2word_set = set(model.index2word)
    for i in file_nouns:
        content_afv = avg_feature_vector(i, model=model, num_features=300,
                                         index2word_set=index2word_set)
        content_vectors.append(content_afv)
    #print content_vectors

    doc_simi_scores = []
    for i in content_vectors:
        lst1 = []
        for j in content_vectors:
            lst1.append(dot(i, j) / (norm(i) * norm(j)))
        doc_simi_scores.append(lst1)
    context_simi_matrix = numpy.array(doc_simi_scores)
    print context_simi_matrix