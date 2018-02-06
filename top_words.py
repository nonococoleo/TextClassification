from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from key_phrases import KeyPhrases
from pickle import dump
from pickle import load


def dump_top_words(model, feature_names, words, s):
    """导出多篇文章的关键词列表"""
    temp = []
    for topic_idx, topic in enumerate(model.components_):
        print(topic_idx, end=" ")
        message = " ".join([feature_names[i] for i in topic.argsort()[:-words - 1:-1]])
        print(message)
        temp.append(message)
    out = open('test_word_%s.pkl' % s, 'wb')
    dump(temp, out, -1)
    out.close()


def get_top_words(model, feature_names, words):
    """返回一篇文章的关联词串，以空格分隔"""
    temp = set()
    for topic_idx, topic in enumerate(model.components_):
        for i in topic.argsort()[:-words - 1:-1]:
            temp.add(feature_names[i])
    return " ".join(temp)


def model_lda(data_samples, n_components, n_top_words=10, n_features=1000, s=None):
    """Fitting LDA models with tf features, n_samples and n_features"""
    # Use tf (raw term count) features for LDA. Extracting tf features for LDA.
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features)
    tf = tf_vectorizer.fit_transform(data_samples)

    # Fitting LDA models with tf features
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online', learning_offset=50.,
                                    random_state=0).fit(tf)

    # Topics in LDA model
    tf_feature_names = tf_vectorizer.get_feature_names()

    if s:
        dump_top_words(lda, tf_feature_names, n_top_words, s)
    return get_top_words(lda, tf_feature_names, n_top_words)


def model_nmf1(data_samples, n_components, n_top_words=10, n_features=1000, s=None):
    """Fitting the NMF model (Frobenius norm) with tf-idf features, n_samples and n_features"""
    # Use tf-idf features for NMF. Extracting tf-idf features for NMF.
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features)
    tfidf = tfidf_vectorizer.fit_transform(data_samples)

    # Fit the NMF model
    nmf = NMF(n_components=n_components, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)

    # Topics in NMF model (Frobenius norm):
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    if s:
        dump_top_words(nmf, tfidf_feature_names, n_top_words, s)
    return get_top_words(nmf, tfidf_feature_names, n_top_words)


def model_nmf2(data_samples, n_components, n_top_words=10, n_features=1000, s=None):
    """Fitting the NMF model (generalized Kullback-Leibler divergence) with tf-idf features, n_samples and n_features"""
    # Use tf-idf features for NMF. Extracting tf-idf features for NMF.
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features)
    tfidf = tfidf_vectorizer.fit_transform(data_samples)

    # Fit the NMF model
    nmf = NMF(n_components=n_components, random_state=1, beta_loss='kullback-leibler',
              solver='mu', max_iter=1000, alpha=.1, l1_ratio=.5).fit(tfidf)

    # Topics in NMF model (generalized Kullback-Leibler divergence)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    if s:
        dump_top_words(nmf, tfidf_feature_names, n_top_words, s)
    return get_top_words(nmf, tfidf_feature_names, n_top_words)


def top_words(text, s=None):
    """输入一段文本，返回这段文本的关键词以空格分隔的字符串"""
    inp = open('data/pre-lda86.6/train_all.pkl', 'rb')
    corpus_all = load(inp)
    inp.close()
    app = KeyPhrases(corpus_all)
    data = []
    corpus = text.split("\n")
    for i in corpus:
        if i:
            app.key_phrases(i)
            data.append(" ".join([" ".join(x) for x in app.phrases]))
    if s == "nmf1":
        return model_nmf1(data, len(data))
    elif s == "nmf2":
        return model_nmf2(data, len(data))
    else:
        return model_lda(data, len(data))


if __name__ == '__main__':
    # 批处理
    inp = open('data/pre-lda86.6/test_all.pkl', 'rb')
    corpus_all = load(inp)
    inp.close()
    app = KeyPhrases(corpus_all)
    for i in ['business', 'enter', 'pol', 'sport', 'tech']:
        # for i in ["athletics", "cricket", "football", "rugby", "tennis"]:
        inp = open('data/pre-lda86.6/test_%s.pkl' % i, 'rb')
        corpus = load(inp)
        inp.close()
        txtNum = len(corpus)  # 语料库的文档总数
        data = []
        print(i)
        for j in range(0, txtNum):
            app.key_phrases(corpus[j])
            temp = " ".join([" ".join(x) for x in app.phrases])
            if temp:
                data.append(temp)
        model_nmf1(data, len(data), s=i)

    # 单段文本
    # text = ''' '''
    # print(top_words(text,"nmf2"))
