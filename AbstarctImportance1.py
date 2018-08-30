import os, json, numpy
import nltk
import enchant
from nltk.corpus import stopwords
import gensim
import string

def onlyascii(char):
    if ord(char) < 48 or ord(char) > 127: return ''
    else: return char

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

'''
getNouns function gets the nouns from the input. It tokenizes and sorted the 
nouns based on their frequency in descending order.
'''
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
    file_abstract_nouns = []
    file_content_nouns = []
    for file in json_files:
        data = json.load(open(path_to_json + str(file)))
        '''
        Getting top 5 nouns of Tika Generated Abstract
        '''
        abstract_nouns,abstract_nouns_freq = getNouns(str(data[0]['AB']))
        file_abstract_nouns.append(abstract_nouns_freq[:4])
        '''
        Getting top 5 nouns of Tika Generated content
        '''
        content_data = filter(lambda x: x in printable, data[0]['X-TIKA:content'])
        content_nouns,content_nouns_freq = getNouns(content_data)
        file_content_nouns.append(content_nouns_freq[:4])

    '''
    To create Gensim model, download GoogleNews-vectors-negative300.bin and provide the path.
    '''
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    index2word_set = set(model.index2word)
    abstract_vectors = []
    content_vectors = []
    '''
    Generate Average word2vec vectors for abstracts and tika generated contents
    '''
    for idx, i in enumerate(file_content_nouns):

        content_afv = avg_feature_vector(file_content_nouns[idx], model=model, num_features=300,
                                         index2word_set=index2word_set)
        content_vectors.append(content_afv)
        abstract_afv = avg_feature_vector(file_abstract_nouns[idx], model=model, num_features=300,
                                          index2word_set=index2word_set)

        abstract_vectors.append(abstract_afv)
    abs = numpy.array(abstract_vectors)
    con = numpy.array(content_vectors)
    out1 = []
    '''
    Prepare content X abstract similarity matrix by using cosine similarity
    '''
    for i in abs:
        out = []
        for j in con:
            sim = dot(i, j) / (norm(i) * norm(j))
            out.append(sim)
        out1.append(out)
    print out1
    exit(0)



