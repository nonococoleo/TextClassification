import nltk
from pickle import dump
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents()
# size = int(len(brown_tagged_sents) * 0.9)
# train_sents = brown_tagged_sents[:size]
# test_sents = brown_tagged_sents[size:]

train_sents = brown_tagged_sents
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
output = open('tagger.pkl', 'wb')
dump(t2, output, -1)
output.close()
# print(t2.evaluate(test_sents))
