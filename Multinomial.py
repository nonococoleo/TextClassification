import math


class Bayes:
    max = 1000000000

    def __init__(self, trainset):
        self.cat_num_docs = dict()
        self.cat_word_dict = dict()
        self.cat_word_count_dict = dict()
        self.trainset = []
        self.vocab_length = 0
        cat = [x[1] for x in trainset]
        for c in set(cat):
            self.cat_num_docs[c] = cat.count(c)
        self.trainset = trainset
        self.train(trainset)

    def train(self, trainset):
        for file in trainset:
            list_words = file[0]
            cat = file[1]

            self.cat_word_dict[cat] = self.cat_word_dict.get(cat, {})
            self.cat_word_count_dict[cat] = self.cat_word_count_dict.get(cat, 0)
            self.cat_word_count_dict[cat] += len(list_words)

            for w in list_words:
                self.cat_word_dict[cat][w] = self.cat_word_dict[cat].get(w, 0)
                self.cat_word_dict[cat][w] += 1
        for dic in self.cat_word_dict.values():
            self.vocab_length += len(dic)

    def classify(self, list_words):
        minimum_neg_log_prob = self.max
        min_category = ''
        length_train = 0
        for cat in self.cat_word_count_dict:
            length_train += self.cat_word_count_dict[cat]
        for cat in self.cat_word_count_dict:
            neg_log_prob = -math.log(self.cat_word_count_dict[cat] / length_train)
            word_dict = self.cat_word_dict[cat]
            count_cat = self.cat_word_count_dict[cat]
            for w in list_words:
                count_word_train = word_dict.get(w, 0)
                ratio = (count_word_train + 1) / (count_cat + self.vocab_length)
                neg_log_prob -= math.log(ratio)

            if minimum_neg_log_prob > neg_log_prob:
                min_category = cat
                minimum_neg_log_prob = neg_log_prob

        return min_category

    def accuracy(self, testset, flag=False):
        ac = 0.0
        all = len(testset)
        recall = dict()
        precision = dict()
        judge = dict()
        for cat in self.cat_num_docs.keys():
            recall[cat] = 0.0
            precision[cat] = 0.0
            judge[cat] = 0.0
        for i in testset:
            cat = self.classify(i[0])
            if cat == i[1]:
                ac += 1
                judge[cat] += 1
            recall[i[1]] += 1
            precision[cat] += 1
        for cat in judge.keys():
            if recall[cat] != 0:
                recall[cat] = judge[cat] / recall[cat]
            if precision[cat] != 0:
                precision[cat] = judge[cat] / precision[cat]
        if flag:
            print("accuracy", ac / all)
            print("recall", recall)
            print("precision", precision)
        return ac / all


if __name__ == "__main__":
    trainset = [(["1", "2", "3"], "pol"), (["5", "4", "3"], "enter")]
    app = Bayes(trainset)
    print(app.classify(["1", "2", "3"]))
