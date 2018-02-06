import collections
import math
from chunker import ConsecutiveNPChunker, ConsecutiveNPChunkTagger
from pickle import load
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from sklearn.datasets import fetch_20newsgroups


class KeyPhrases:
    def __init__(self, text):
        """初始化。存储文章，记录文章数，标注词性和词组，分割单词和短语"""
        inp = open('data/tagger.pkl', 'rb')  # 载入训练好的词性标注器
        tagger = load(inp)
        inp.close()
        inp = open('data/chunker.pkl', 'rb')  # 载入训练好的实体标注器
        chunker = load(inp)
        inp.close()
        self.tagger = tagger
        self.chunker = chunker
        self.corpus = text
        self.textnum = len(text)

    def tag(self, text):
        """分割词语，标注词性和实体，标注好词性的短语列表的列表"""
        tokens = nltk.sent_tokenize(text)  # 按句分割['sent1','sent2'...]

        tagged_sents = []  # 标注好词性的句子[[('word1','tag1'),('word2','tag2')...],[('word1','tag1'),('word2','tag2')...]...]
        for token in tokens:
            sent = WordPunctTokenizer().tokenize(token)  # 按词分割
            temp = self.tagger.tag(sent)  # 标注
            tagged_sents.append(temp)  # 存储

        tagged_phrases = []  # 标注好词性的短语列表的列表[[('word1','tag1'),('word2','tag2')],[('word1','tag1')]...]
        for token in tagged_sents:
            result = self.chunker.parse(token)  # 标注
            nps = result.subtrees((lambda i: i.label() == 'NP' or i.label() == 'VP'))  # 筛选名词、动词性短语
            for np in nps:
                temp = []  # 单个短语
                for word in np.pos():
                    temp.append(word[0])
                if temp:
                    tagged_phrases.append(temp)  # 添加
        return tagged_phrases

    def clean_phrases(self):
        """清理短语，返回不带词性的短语元组的集合"""
        tagged_phrases = self.tagged_phrases
        phrases = []  # 短语元组的列表[('word1','word2'),('word1','word2','word3')...]
        words = [] # 单词的集合，用于tfidf中计算词频
        stop_words = stopwords.words('english')
        stemmer = SnowballStemmer("english")
        for phrase in tagged_phrases:
            temp = []  # 短语列表['word1','word2','word3']
            for tagged_word in phrase:
                word = stemmer.stem(tagged_word[0])  # 词干化
                if wordnet.synsets(word) and len(word) > 1 and word not in stop_words and not word.isnumeric():  # 清理词
                    temp.append(word)
                    words.append(word)
            if temp:
                phrases.append(tuple(temp))  # 转换为元组存储
        self.words = words
        return set(phrases)

    def get_tfidf(self):
        """计算词的权重，返回权重字典"""
        tfidf = dict()  # 每个词tf-idf值的字典
        words = self.words
        wordsnum = len(words)  # 当前文本的总词数
        fre = collections.Counter(words)  # 单词频率统计
        words = list(fre.keys())
        for word in words:
            wordtf = float(fre[word]) / float(wordsnum)  # 计算一个词的tf值
            contain = 0  # 统计包含该词的文档总数
            for i in self.corpus:
                if word in i:
                    contain += 1
            wordidf = math.log2(float(self.textnum + 1) / float(contain + 1))  # 计算一个词的idf值
            temp = wordtf * wordidf  # 计算一个词的tfidf值
            tfidf[word] = temp  # 存储
        return tfidf

    def print_key_phrases(self, x=20, flag=False, show=False):
        """打印前x个关键词组,（不）带权重值"""
        wordtfidf = self.get_tfidf()
        key = []
        phrases_list = list(self.phrases)
        phrases = []
        words = set()
        for i in self.phrases:
            phrases.append(" ".join(i))
        for i in range(len(phrases_list)):  # 词组的权重由里面的每个单词权重相加得到
            temp = 0
            for j in phrases_list[i]:
                temp += wordtfidf[j]
                words.add(j)
            if temp > 0:
                key.append((phrases[i], temp))
        key = sorted(key, key=lambda x: x[1], reverse=True)  # 按权重从大到小排序
        if show:
            for i in key[:x]:  # 打印
                if flag:
                    print(i[0], i[1])
                else:
                    print(i[0])
        return words

    def key_phrases(self, text, x=None, flag=None, show=None):
        self.tagged_phrases = self.tag(text)
        self.phrases = self.clean_phrases()
        return self.print_key_phrases(x, flag, show)


if __name__ == '__main__':
    text = ''' Federer claims Dubai crown

World number one Roger Federer added the Dubai Championship trophy to his long list of successes - but not before he was given a test by Ivan Ljubicic.

Top seed Federer looked to be on course for a easy victory when he thumped the eighth seed 6-1 in the first set. But Ljubicic, who beat Tim Henman in the last eight, dug deep to secure the second set after a tense tiebreak. Swiss star Federer was not about to lose his cool, though, turning on the style to win the deciding set 6-3. The match was a re-run of last week's final at the World Indoor Tournament in Rotterdam, where Federer triumphed, but not until Ljubicic had stretched him for five sets. "I really wanted to get off to a good start this time, and I did, and I could really play with confidence while he still looking for his rhythm," Federer said.

"That took me all the way through to 6-1 3-1 0-30 on his serve and I almost ran away with it. But he came back, and that was a good effort on his side." Ljubicic was at a loss to explain his poor showing in the first set. "I didn't start badly, but then suddenly I felt like my racket was loose and the balls were flying a little bit too much. And with Roger, if you relax for a second it just goes very quick," he said. "After those first three games it was no match at all. I don't know, it was really weird. I was playing really well the whole year, and then suddenly I found myself in trouble just to put the ball in the court." But despite his defeat, the world number 14 was pleased with his overall performance. "I had a chance in the third, and for me it's really positive to twice in two weeks have a chance against Roger to win the match. "It's an absolutely great boost to my confidence that I'm up there and belong with top-class players."
'''  # 要处理的文章
    inp = open('data/corpus/bbc/train_all.pkl', 'rb')  # 训练文集
    corpus = load(inp)
    inp.close()
    app = KeyPhrases(corpus)  # 构造文章处理器
    app.key_phrases(text, 20, True, True)  # 打印关键短语

    # 批量处理
    # inp = open('data/pre-lda86.6/train_all.pkl', 'rb')
    # corpus_all = load(inp)
    # inp.close()
    # temp = []
    # app = KeyPhrases(corpus_all)
    # for i in ['business', 'enter', 'pol', 'sport', 'tech']:
    #     # for i in ["athletics", "cricket", "football", "rugby", "tennis"]:
    #     inp = open('data/pre-lda86.6/train_%s.pkl' % i, 'rb')
    #     corpus = load(inp)
    #     inp.close()
    #     textnum = len(corpus)  # 语料库的文档总数
    #     for j in range(1, textnum):
    #         print(j)
    #         temp.append((app.key_phrases(corpus[j], 20, True,True), i))
    # out = open('train_word_all.pkl', 'wb')
    # dump(temp, out, -1)
    # out.close()

    # li=["alt.atheism","comp.graphics","comp.os.ms-windows.misc","comp.sys.ibm.pc.hardware","comp.sys.mac.hardware","comp.windows.x","misc.forsale","rec.autos","rec.motorcycles","rec.sport.baseball","rec.sport.hockey","sci.crypt","sci.electronics","sci.med","sci.space","soc.religion.christian","talk.politics.guns","talk.politics.mideast","talk.politics.misc","talk.religion.misc"]
    # corpus_all= fetch_20newsgroups(subset='test', categories=li, shuffle=True, random_state=42).data
    # temp=[]
    # app = KeyPhrases(corpus_all)
    # for i in li:
    #     corpus = fetch_20newsgroups(subset='test', categories=[i], shuffle=True, random_state=42).data
    #     textnum = len(corpus)  # 语料库的文档总数
    #     for j in range(1, textnum):
    #         print(j)
    #         app.start(corpus[j], textnum)
    #         temp.append((app.print_key_phrases(20, True), i))
    # out = open('20_test_word_all.pkl', 'wb')
    # dump(temp, out, -1)
    # out.close()
