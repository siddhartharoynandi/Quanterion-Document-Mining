import os, json, numpy
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
import math
import enchant
from collections import defaultdict
from sklearn.decomposition import PCA
from skfuzzy.cluster import cmeans
from sklearn.cluster import KMeans
import hdbscan
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt

'''
This model takes all unique words from tika generated abstracts. Generates word2vec vectors 
from each word. It builds Bag-Of-Concepts model, where concepts are columns and documents are
rows. Different clustering techniques and calculations are done.Currentl fuzzy clustering section 
is active.Rest of clustering(k-means,HDBSCAN) commented.
'''

############################## Tokenizing Abstracts ##############################################
def cleanAbstract(file_abstract_data):
    clean_abstract_data = []
    d = enchant.Dict("en_US")
    for i in file_abstract_data:
        for j in i:
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(j.lower())
            words = [w for w in tokens if not w in stopwords.words('english')]
            vocab_words = [w for w in words if not w.isdigit()]
            vocab_words2 = [w for w in vocab_words if d.check(w)]
            print set(vocab_words2)
            clean_abstract_data.append(vocab_words2)
    return clean_abstract_data

############################# Create Set of Unique Words ########################################
def createUniqueWordSet(clean_abstract_data):
    clean_abstract_sets_list = []
    for i in clean_abstract_data:
        clean_abstract_sets_list.append(set(i))
    # print(len(clean_abstract_set))
    unique_words = set()
    for i in clean_abstract_sets_list:
        unique_words = unique_words | i
    #print unique_words
    return unique_words

############################# Create data points from word2vec ###################################
def createWordVectors(unique_words):
    word_data_points = []
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    index2word_set = set(model.index2word)
    #Cleaning Words from Unique Word list which are not present in index2word_set
    remove_words = []
    for i in unique_words:
        if i in index2word_set:
            continue
        else:
            remove_words.append(i)
    for i in remove_words:
        unique_words.remove(i)
    #Generating word vectors
    for j in unique_words:
        word_data_points.append(model.wv[j])
    return word_data_points,unique_words

################# Create a list for each document where tuples are word frequency ###############
def createDocumentFrequencyList(unique_words,clean_abstract_data):
    num_unique_words = len(unique_words)
    doc_word_count = []
    for docdata in clean_abstract_data:  # iterating over list of lists
        doc_word_frquency = numpy.zeros(num_unique_words)
        for idx, tokens in enumerate(unique_words):
            token_count = docdata.count(tokens)
            doc_word_frquency[idx] = token_count
        doc_word_count.append(doc_word_frquency)
    return doc_word_count

############################# PCA and Fuzzy C-means Section #####################################
def dimensionReductionandClustering(word_data_points):
    word_data_array = numpy.array(word_data_points)
    pca = PCA(n_components=2)
    result = pca.fit_transform(word_data_array)
    result_T = result.transpose()
    k = 6
    cntr, mem_matr, u0, d, jm, p, fpc = cmeans(result_T, k, 2, error=0.005, maxiter=1000, init=None)
    return mem_matr,k

############################# Generate a dictionary for each word maintaing indeces of the clusters #######################
def createClusterWordDictionary(membership_matrix):
    dictclusterpoints = defaultdict(list)
    for i in range(membership_matrix.shape[1]):
        for j in range(membership_matrix.shape[0]):
            if membership_matrix[j][i] > 0.0005:
                dictclusterpoints[i].append(j)
    return dictclusterpoints

############################# Generate a dictionary for each cluster maintaing indeces of the words #######################
def createWordClusterDictionary(membership_matrix):
    dictdatapoints = defaultdict(list)
    for i in range(membership_matrix.shape[0]):
        for j in range(membership_matrix.shape[1]):
            if membership_matrix[i][j] > 0.0005:
                dictdatapoints[i].append(j)
    return dictdatapoints

########################### Create Document X Concept matrix by CF*IDF*FuzzyScore #####################
def createDocumentConceptMatrix(num_documents,num_concepts,doc_word_count,dictdatapoints,membership_matrix):
    document_concept_matrix = numpy.zeros((num_documents, num_concepts))
    #Filling up the Concept Frequency * Fuzzy Score
    for i in range(num_documents):
        for j in range(num_concepts):
            doc_words_index = []
            for idx,value in enumerate(doc_word_count[i]):
                if value != 0:
                    doc_words_index.append(idx)
            common_word_index = list(set(doc_words_index).intersection(set(dictdatapoints[j])))
            for id in common_word_index:
                document_concept_matrix[i][j] = doc_word_count[i][id] * membership_matrix[j][id]
            document_concept_matrix[i][j] = document_concept_matrix[i][j] / sum(doc_word_count[i])
    #print(document_concept_matrix)
    #Calculate the IDF values
    '''
    IDF_list = []
    for j in range(num_concepts):
        count = 0
        for i in range(num_documents):
            if document_concept_matrix[i][j] != 0:
                count += 1
        IDF_list.append(count)
    print(IDF_list)
    for i in range(num_documents):
        for j in range(num_concepts):
            document_concept_matrix[i][j] = document_concept_matrix[i][j] * math.log((num_documents / (IDF_list[j])),10)
    '''
    return document_concept_matrix



if __name__ == '__main__':
    path_to_json = '/Users/siddhartharoynandi/Desktop/RA_2018/new_data_json/'
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    print(json_files)
    num_documents = len(json_files)
    file_abstract_data = []
    for file in json_files:
        file_data = []
        data = json.load(open(path_to_json + str(file)))
        file_data.append(str(data[0]['AB']))
        file_abstract_data.append(file_data)
    clean_abstract_data = cleanAbstract(file_abstract_data)
    unique_words = createUniqueWordSet(clean_abstract_data)
    word_data_points,unique_words2 = createWordVectors(unique_words)
    doc_word_count = createDocumentFrequencyList(unique_words2,clean_abstract_data)
    membership_matrix,num_concepts = dimensionReductionandClustering(word_data_points)
    #print(membership_matrix)
    #dictclusterpoints = createClusterWordDictionary(membership_matrix)
    dictdatapoints = createWordClusterDictionary(membership_matrix)
    #print(dictclusterpoints)
    document_concept_matrix = createDocumentConceptMatrix(num_documents,num_concepts,doc_word_count,dictdatapoints,membership_matrix)
    #print(document_concept_matrix)
    #dot_matrix = numpy.dot(document_concept_matrix,document_concept_matrix.T)
    #print(dot_matrix)
    pca = PCA(n_components=5)
    result = pca.fit_transform(document_concept_matrix)
    '''
    kmeans = KMeans(n_clusters=4, max_iter=1000).fit(result)
    print('Kmeans Cluster Labels: ' +str(kmeans.labels_))
    LABEL_COLOR_MAP = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'yellow'}
    color_list = []
    for i in kmeans.labels_:
        color_list.append(LABEL_COLOR_MAP[i])
    plt.scatter(result[:, 0], result[:, 1], c=color_list)
    plt.show()
    exit(0)
    '''
    coordinates = []
    for i in result:
        coordinates.append(i)
    print coordinates
    clusterer = hdbscan.HDBSCAN().fit(result)
    Z = clusterer.single_linkage_tree_.to_numpy()
    doc_labels = fcluster(Z, 4, criterion='maxclust')
    print('HDBSCAN Cluster Labels: '+str(doc_labels))
    LABEL_COLOR_MAP = {1: 'red', 2: 'blue', 3: 'green', 4: 'purple', 5: 'yellow'}
    color_list = []
    for i in doc_labels:
        color_list.append(LABEL_COLOR_MAP.get(i))
    plt.scatter(result[:, 0], result[:, 1], c=color_list)
    plt.show()
    exit(0)

