import os, json, numpy
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
import enchant
from collections import defaultdict
import math
from sklearn.decomposition import PCA
import seaborn as sns
from skfuzzy.cluster import cmeans
from sklearn.cluster import KMeans
import hdbscan
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt

'''
This model generates average word vectors for each abstarct and apply
clustering to see the concept clusters. Optimal number of clusters are 
determined by elbow method(commented).
'''

def generateAverageVectorsforAbstract(sentence, model, num_features, index2word_set):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    words = [w for w in tokens if not w in stopwords.words('english')]
    #words  = sentence.split()
    #print(words)
    feature_vec = numpy.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in tokens:
        if word in index2word_set:
            n_words += 1
            feature_vec = numpy.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = numpy.divide(feature_vec, n_words)
    return feature_vec

if __name__ == '__main__':
    path_to_json = '/Users/siddhartharoynandi/Desktop/RA_2018/new_data_json/'
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    print(json_files)
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    index2word_set = set(model.index2word)
    file_abstract_data = []
    for file in json_files:
        data = json.load(open(path_to_json + str(file)))
        file_abstract_data.append(generateAverageVectorsforAbstract(str(data[0]['AB'].lower()),model=model, num_features=300, index2word_set=index2word_set))

    pca = PCA(n_components=2)
    result = pca.fit_transform(file_abstract_data)

    '''
    sse = {}
    X = numpy.array(result)
    for k in range(1, 5):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        sse[k] = kmeans.inertia_

    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()
    exit(0)
    '''
    coordinates = []
    for i in result:
        coordinates.append(i)
    print coordinates
    clusterer = hdbscan.HDBSCAN().fit(result)
    Z = clusterer.single_linkage_tree_.to_numpy()
    doc_labels = fcluster(Z, 5, criterion='maxclust')
    print('HDBSCAN Cluster Labels: ' + str(doc_labels))
    LABEL_COLOR_MAP = {1: 'red', 2: 'blue', 3: 'green', 4: 'purple', 5: 'yellow'}
    color_list = []
    for i in doc_labels:
        color_list.append(LABEL_COLOR_MAP.get(i))
    plt.scatter(result[:, 0], result[:, 1], c=color_list)
    plt.show()
    exit(0)