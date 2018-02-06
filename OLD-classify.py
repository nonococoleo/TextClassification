from pickle import load
import random
from chunker import *
from top_words import top_words
import matplotlib.pyplot as pt


def feature(tokens):
    """构造特征字典"""
    dic = dict()
    for word in tokens.split():
        dic["have(%s)" % word] = True
    return dic


train_set = []  # 构造训练集
for i in ['business', 'enter', 'pol', 'sport', 'tech']:
    # for i in ["athletics", "cricket", "football", "rugby", "tennis"]:
    inp = open('data/pre-lda86.6/train_word_%s.pkl' % i, 'rb')  # 载入分类文档
    t = load(inp)
    inp.close()

    for line in t:
        train_set.append((feature(line), i))  # 处理成关键词和标签的字典
random.shuffle(train_set)
print("train set loaded")

classifier = nltk.NaiveBayesClassifier.train(train_set)  # 训练分类器
print("classifier built")

# 分类器正确率评估
test_set = []  # 构造测试集
for i in ['business', 'enter', 'pol', 'sport', 'tech']:
    # for i in ["athletics", "cricket", "football", "rugby", "tennis"]:
    inp = open('data/pre-lda86.6/test_word_%s.pkl' % i, 'rb')  # 载入分类文档
    t = load(inp)
    inp.close()
    for line in t:
        test_set.append((feature(line), i))  # 处理成关键词和标签的字典

random.shuffle(test_set)
print("test set loaded")
print("accuracy:", nltk.classify.accuracy(classifier, test_set) * 100, "%")  # 输出正确率

# 单文本标记
news = ''' '''  # 要打标签的文本
print(classifier.classify(feature(top_words(news))))

# 正确率影响因素研究
features = train_set
random.shuffle(features)
si = []
acc = []
for i in range(1, 40, 1):
    proportion = i / 100
    size = int(len(features) * proportion)
    train_set, test_set = features[size:], features[:size]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    acc.append(nltk.classify.accuracy(classifier, test_set))
    si.append(proportion)
pt.plot(si, acc)
pt.show()
