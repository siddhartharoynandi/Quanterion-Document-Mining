import os, json, numpy
import nltk
import enchant
from nltk.corpus import wordnet
import gensim
from scipy import spatial

'''
getNouns returns top 5 nouns from the input abstract words.
'''
def getNouns(abstract):
    d = enchant.Dict("en_US")
    sentences = nltk.sent_tokenize(abstract)  # tokenize sentences
    nouns = []  # empty to array to hold all nouns
    frequency_nouns = {}
    for sentence in sentences:
        for word, pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                if d.check(word):
                    nouns.append(word)
    for _word in nouns:
        if _word in frequency_nouns:
            frequency_nouns[_word] += 1
        else:
            frequency_nouns[_word] = 1
    sorted_keys = sorted(frequency_nouns, key=frequency_nouns.get, reverse=True)
    #print frequency_nouns
    return sorted_keys[:6]
'''
wordnetSimilarity calculates normalized taxonomic similarity
between 2 list of nouns which are representatitive of two
documents.
'''

def wordnetSimilarity(list1, list2):
    #print(list1, list2)
    list = list11 = []
    if list1 == list2:
        return 1.0
    for word1 in list1:
        k = []
        for word2 in list2:
            #print word1, word2
            wordFromList1 = wordnet.synsets(word1)
            wordFromList2 = wordnet.synsets(word2)
            if wordFromList1 and wordFromList2:
                s = wordFromList1[0].wup_similarity(wordFromList2[0])
                if (str(s) == 'None'): s = 0.0
                k.append(s)
                #print k
        if len(k) > 0:
            list.append(max(k))

    for word1 in list2:
        k = []
        for word2 in list1:
            #print word1, word2
            wordFromList1 = wordnet.synsets(word1)
            wordFromList2 = wordnet.synsets(word2)
            if wordFromList1 and wordFromList2:
                s = wordFromList1[0].wup_similarity(wordFromList2[0])
                if (str(s) == 'None'): s = 0.0
                k.append(s)
                #print k
        if len(k) > 0:
            list11.append(max(k))

    return (sum(list) + sum(list11)) / (len(list) + len(list11))

'''
word2VecSimilarity calculates contextual similarity between two list 
of word vectors which are representatitive of two documents.
'''

def word2VecSimilarity(list1,list2):
    list = list11 = []
    for word1 in list1:
        k = []
        for word2 in list2:
            k.append(1 - spatial.distance.cosine(word1, word2))
        list.append(max(k))

    for word1 in list2:
        k = []
        for word2 in list1:
            k.append(1 - spatial.distance.cosine(word2, word1))
        list11.append(max(k))
    return (sum(list) + sum(list11)) / (len(list) + len(list11))

if __name__ == '__main__':
    path_to_json = '/Users/siddhartharoynandi/Desktop/RA_2018/new_data_json/'
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    print json_files
    file_nouns = []
    for file in json_files:
        data = json.load(open(path_to_json + str(file)))
        file_nouns.append(getNouns(str(data[0]['AB'])))
    print file_nouns
    exit(0)
    simi_scores = []
    for i in file_nouns:
        lst = []
        for j in file_nouns:
            lst.append(wordnetSimilarity(i,j))
        simi_scores.append(lst)
    tax_simi_matrix = numpy.array(simi_scores)
    print tax_simi_matrix

    ########## Word2Vector Similarity Section ############

    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    index2word_set = set(model.index2word)
    wordvectors = []
    for i in file_nouns:
        lst1 = []
        for j in i:
            if j in index2word_set:
                lst1.append(model.wv[j])
            else:
                lst1.append([0.0]*300)
        wordvectors.append(lst1)
    #print wordvectors

    doc_simi_scores = []
    for i in wordvectors:
        lst1 = []
        for j in wordvectors:
            lst1.append(word2VecSimilarity(i,j))
        doc_simi_scores.append(lst1)
    context_simi_matrix = numpy.array(doc_simi_scores)
    print context_simi_matrix
    '''
    Final similarity matrix calculated by linear equation of weighted 
    taxonomic and contextual similarity. Weights can be changed accordingly.
    '''
    ####### Final Similarity = 0.5 * Taxonomical Similarity + 0.5 * Contextual Similarity ############
    final_similarity = numpy.zeros((len(json_files),len(json_files)))
    for i in range(0,final_similarity.shape[0]):
        for j in range(0,final_similarity.shape[0]):
            final_similarity[i][j] = 1.0*tax_simi_matrix[i][j] + 0.0*context_simi_matrix[i][j]
    print final_similarity

    exit(0)