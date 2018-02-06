from pickle import load
import random
from Multinomial import Bayes
# from Bernoulli import Bayes
from key_phrases import KeyPhrases
from chunker import ConsecutiveNPChunker, ConsecutiveNPChunkTagger
import matplotlib.pyplot as pt
from time import time

t0 = time()
train_set = []  # 构造训练集
inp = open('data/train_word_all.pkl', 'rb')  # 载入分类文档
t = load(inp)
inp.close()
for line in t:
    if line[0]:
        train_set.append((set(line[0]), line[1]))
random.shuffle(train_set)
print("train set loaded", len(train_set))
print("done in %0.3fs." % (time() - t0))

t0 = time()
app = Bayes(train_set)
print("classifier built")
print("done in %0.3fs." % (time() - t0))

# 分类器正确率评估
# t0 = time()
# test_set = []  # 构造测试集
# inp = open('data/sport_test_word_all.pkl', 'rb')  # 载入分类文档
# t = load(inp)
# inp.close()
# for line in t:
#     if line[0]:
#         test_set.append((set(line[0]), line[1]))
# random.shuffle(test_set)
# print("test set loaded", len(test_set))
# print("done in %0.3fs." % (time() - t0))
#
# t0 = time()
# app.accuracy(test_set, True)  # 输出正确率
# print("done in %0.3fs." % (time() - t0))

# 单文本标记
t0 = time()
news = '''The boss of Goldman Sachs has warned that the US bank's contingency planning is reaching the point of no return.
The bank's chief executive, Lloyd Blankfein, told the BBC some steps already taken to deal with Brexit were now very unlikely to be reversed.
At some point the steps are "not going to be undone", Mr Blankfein said.
For example, some contracts between Goldman Sachs UK and EU clients have been redrawn, or "repapered", to apply to Goldman Sachs Germany.
"Once we start to repaper - which is very cumbersome because it involves lots of lawyers on both sides and takes months - once we start that are we going to go back? Probably not," Mr Blankfein told the BBC at the World Economic Forum in Davos.
"We've already done some and we have told our clients that more is coming," he said.
'Can't be undone'
When asked exactly when the final point of no return would arrive, he said it was not a binary thing but a gradual process. '''  # 要打标签的文本
inp = open('data/corpus/bbc/train_all.pkl', 'rb')
corpus_all = load(inp)
inp.close()
phrases = KeyPhrases(corpus_all)
print("catagory:",app.classify(set(phrases.key_phrases(news))))
print("done in %0.3fs." % (time() - t0))

# 正确率影响因素研究
# features = train_set
# si = []
# acc = []
# for i in range(1, 101, 2):
#     proportion = i / 100
#     size = int(len(features) * proportion)
#     random.shuffle(features)
#     train, test = features[size:], features[:size]
#     acc.append(Bayes(train).accuracy(test))
#     si.append(proportion)
# pt.plot(si, acc)
# pt.show()
