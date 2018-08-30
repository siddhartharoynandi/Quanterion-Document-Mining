
import os, PyPDF2
import nltk
import enchant
from nltk.corpus import wordnet
import string
from operator import itemgetter
'''
getNouns function takes the content of the pdfs as a list one by one.
It then tokenize the content and identify the nouns.Creates a dictionary
of nouns with frequencies. It returns the sorted dictionary based on the
frequency of nouns in descending order.
'''
def getNouns(abstract):
    d = enchant.Dict("en_US")
    sentences = nltk.sent_tokenize(abstract.decode('utf-8'))  # tokenize sentences
    nouns = []  # empty to array to hold all nouns
    frequency_nouns = {}
    for sentence in sentences:
        for word, pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                if d.check(word.lower()):
                    nouns.append(word.lower())
    for _word in nouns:
        if _word in frequency_nouns:
            frequency_nouns[_word] += 1
        else:
            frequency_nouns[_word] = 1
    #print frequency_nouns
    sorted_values = sorted(frequency_nouns.items(), key=itemgetter(1),reverse= True)
    #print sorted_values
    return sorted_values

'''
mergeNouns function takes the sorted dictionary as input. It merges all the nouns which
have similarity more than 0.86 and add up their frequencies.The representitive noun is
the more frequent noun among two which are going to merge. Once all the nouns are merged
they are sorted in descending order by their frequency.
'''
def mergeNouns(content_nouns_freq):
    dict = {}
    for i in content_nouns_freq:
        #print i[0]
        first_word = i[0]
        dict[first_word] = i[1]
        wordFromList1 = wordnet.synsets(first_word)
        #print wordFromList1
        for j in content_nouns_freq[::-1]:
            if i[0] != j[0] and j[0] not in dict:
                #print j[0]
                second_word = j[0]
                wordFromList2 = wordnet.synsets(second_word)
                # print wordFromList1[0].wup_similarity(wordFromList2[0])
                if wordFromList1 != [] and wordFromList2 != []:
                    if wordFromList1[0].wup_similarity(wordFromList2[0]) > 0.86:
                        dict[first_word] += j[1]
                        content_nouns_freq.remove(j)
    return dict

'''
searchResults function is for the searching part. Each file is represented by
sorted merged nouns. searchResults searches for noun's index in the representative
merged noun structures. First it identifies nouns in the search query, then it 
looks for similar nouns indices in the merged noun structures. If it finds a match
it adds the noun's position in the merged noun representation of that file and the
filename.
'''

def searchResults(key_words,search_string,file_index):
    search_noun_keywords = []
    document_keyword_indices = []
    d = enchant.Dict("en_US")
    for word, pos in nltk.pos_tag(nltk.word_tokenize(str(search_string))):
        if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
            if d.check(word.lower()):
                search_noun_keywords.append(word.lower())
    for noun in search_noun_keywords:
        wordFromList1 = wordnet.synsets(noun)
        for idx,i in enumerate(key_words):
            for idx2,j in enumerate(i):
                wordFromList2 = wordnet.synsets(j[0])
                if wordFromList1 != [] and wordFromList2 != []:
                    if wordFromList1[0].wup_similarity(wordFromList2[0]) > 0.86:
                        doc_name = file_index[idx]
                        document_keyword_indices.append((idx2,doc_name))
                        break
    return document_keyword_indices




if __name__ == '__main__':
    printable = set(string.printable)
    path_to_pdf = '/Users/siddhartharoynandi/Desktop/RA_2018/TestData/BlockChain/'
    pdf_files = [pos_pdf for pos_pdf in os.listdir(path_to_pdf) if pos_pdf.endswith('.pdf')]
    file_index = {}
    for idx,i in enumerate(pdf_files):
        file_index[idx] = i
        #print json_files
    print file_index
    file_nouns = []
    key_words = []
    for file in pdf_files:
        # Read the pdf files from the given directory location
        pdf_file = open(path_to_pdf + str(file), 'rb')
        read_pdf = PyPDF2.PdfFileReader(pdf_file)
        number_of_pages = read_pdf.getNumPages()
        page_content = ""
        for i in range(number_of_pages):
            page_content += read_pdf.getPage(i).extractText() + "\n"
        page_content = " ".join(page_content.replace(u"\xa0", " ").strip().split())
        content_data = filter(lambda x: x in printable,page_content)
        content_nouns_freq = getNouns(content_data)
        sorted_values = sorted(mergeNouns(content_nouns_freq).items(), key=itemgetter(1),reverse= True)
        #print sorted_values
        key_words.append(sorted_values)
        #print sorted_values

    search_string = 'block'
    print searchResults(key_words,search_string,file_index)
    exit(0)
