import math


class Bayes:
    max = 1000000000

    def __init__(self, trainset):
        self.cat_num_docs = dict()
        self.word_cat_num_doc_dict = dict()
        self.trainset = []
        cat = [x[1] for x in trainset]
        for c in set(cat):
            self.cat_num_docs[c] = cat.count(c)
        self.trainset = trainset
        self.train(trainset)

    def train(self, trainset):
        for file in trainset:
            list_words = file[0]
            cat = file[1]

            for w in set(list_words):
                self.word_cat_num_doc_dict[w] = self.word_cat_num_doc_dict.get(w, {})
                self.word_cat_num_doc_dict[w][cat] = self.word_cat_num_doc_dict[w].get(cat, 0)
                self.word_cat_num_doc_dict[w][cat] += 1

        for w in self.word_cat_num_doc_dict:
            for cat in self.cat_num_docs:
                nct = self.word_cat_num_doc_dict[w].get(cat, 0)
                ratio = (nct + 1) / (self.cat_num_docs[cat] + 2)
                self.word_cat_num_doc_dict[w][cat] = ratio

    def classify(self, set_list_words):
        minimum_neg_log_prob = self.max
        min_category = ''

        for cat in self.cat_num_docs:
            neg_log_prob = -math.log(self.cat_num_docs[cat] / len(self.trainset))
            for w in self.word_cat_num_doc_dict:
                if w in set_list_words:
                    neg_log_prob -= math.log(self.word_cat_num_doc_dict[w][cat])
                else:
                    neg_log_prob -= math.log(1 - self.word_cat_num_doc_dict[w][cat])
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
