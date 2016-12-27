import os, string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import cPickle


def getFeature():
    with open(os.path.join('spam_filter_train.txt'), 'r') as f:
        trainData = f.readlines()
    with open(os.path.join('spam_filter_test.txt'), 'r') as f:
        testData = f.readlines()
    data = trainData + testData
    trainNum, testNum = len(trainData), len(testData)
    del trainData
    del testData

    for i in range(len(data)):
        data[i] = data[i].replace('\n', '').split('\t')[1]

    result = []
    for line in data:
        new_line = string.lower(line)
        new_line = new_line.replace('<br />', ' ')
        new_line = new_line.replace('n\'t', ' not')
        new_line = new_line.replace('he\'s', 'he is')
        new_line = new_line.replace('she\'s', 'she is')
        new_line = new_line.replace('\'m', ' am')
        new_line = new_line.replace('\'re', ' are')
        new_line = new_line.replace('\'ll', ' will')
        new_line = new_line.replace('\'d', ' would')
        table = string.maketrans("","")
        new_line = new_line.translate(table, string.punctuation)
        result.append(new_line)
    # lemmatize
    lemmatized = []
    wnl = WordNetLemmatizer()
    for line in result:
        lemmatized.append([wnl.lemmatize(word) for word in line.split(' ')])
    # stem
    porter_stemmer = PorterStemmer()
    stemmed = []
    for line in lemmatized:
        stemmed.append([porter_stemmer.stem(word) for word in line])
    # remove stopwords
    stopwordRemoved = []
    sw = set(stopwords.words('english'))
    for line in stemmed:
        stopwordRemoved.append(' '.join([x for x in line if x not in sw]))
    # tf feature
    vec = CountVectorizer()
    features = vec.fit_transform(stopwordRemoved)

    with open('trainFeatures.pkl', 'wb') as f:
        cPickle.dump(features[:trainNum], f)
    with open('testFeatures.pkl', 'wb') as f:
        cPickle.dump(features[trainNum:], f)

def main():
    getFeature()
    '''
    with open('trainFeatures.pkl', 'rb') as f:
         trainFeatures = cPickle.load(f)
    with open('testFeatures.pkl', 'rb') as f:
         testFeatures = cPickle.load(f)
    '''


if __name__ == '__main__':
    main()
