#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import jieba
import json
import re
import sys
import numpy as np
from sklearn.model_selection import train_test_split
dirPrefix = "/mnt/hgfs/share/pornCensor/"

class DataLoader(object):
    def __init__(self):
        self.w2v = {}
        self.w2index = {}
        self.weights = []
        self.word_vec_len = 20
        self.ratio = 0.1
        self.title_len = 5
        self.article_len = 10

    def set_w2v(self, path='../../xx'):
        with open(path, "r", errors='ignore') as f:
            index = 0
            for line in f:
                units = line.strip().split()
                if (0 == index):
                    self.word_vec_len = int(units[1])
                    self.w2index['[PAD]'] = index
                    self.weights.append([0.0] * self.word_vec_len) # pad
                else:
                    token = units[0]
                    vec = [float(x) for x in units[1:]]
                    if (len(vec) == self.word_vec_len):
                        self.weights.append(vec)
                        self.w2index[token] = index
                        self.w2v[token] = vec
                    else:
                        continue
                index += 1
            print (index)
    def get_weights(self):
        return np.array(self.weights)

    def save_dict(self, filepath):
        with open(filepath, "w") as fp:
            d = sorted(self.w2index.items(), key=lambda item:item[1])
            for key, value in d:
                fp.write(key+"\n")
#            json.dump(self.w2index, fp)

    def load_dict(self, filepath):
        with open(filepath, "r") as fp:
            index = 0
            for key in fp:
                self.w2index[key] = index
                index += 1
#            self.w2index = json.load(fp)

    def build(self, conf_file):
        with open(conf_file, "r") as files:
            self.title_list = []
            self.Y_list = []
            self.article_list = []
            idx = 0
            for line in files:
                id, filename = line.strip().split()
                print(id, filename)
                counter = 0
                with open(filename, 'r') as sample_file:
                    print(filename,"open success!!!")
                    for line in sample_file:
                        counter += 1
                        try:
                            line = line.strip().lower()
                            title, article = line.split("\t")
                        except Exception as e:
                            print ("error", counter)
                            continue

                        ''' title '''
                        feat_title = []
                        words = []
                        wordsplit = jieba.cut(title, cut_all=False, HMM=False)
                        for qseg in wordsplit:
                            qseg = re.sub(r'\\s+', ' ', qseg).strip()
                            if qseg:
                                try:
                                    feat_title.append(self.w2index[qseg])
                                    words.append(qseg)
                                except KeyError as e:
                                    continue

                        grams = self.get_trigram(title.replace(' ', ''))
                        for gram in grams:
                            if gram in words:
                                continue
                            try:
                                feat_title.append(self.w2index[gram])
                                words.append(gram)
                            except KeyError as e:
                                continue

                        grams = self.get_bigram(title.replace(' ', ''))
                        for gram in grams:
                            if gram in words:
                                continue
                            try:
                                feat_title.append(self.w2index[gram])
                                words.append(gram)
                            except KeyError as e:
                                continue

                        grams = self.get_unigram(title.replace(' ', ''))
                        for gram in grams:
                            if gram in words:
                                continue
                            try:
                                feat_title.append(self.w2index[gram])
                                words.append(gram)
                            except KeyError as e:
                                continue

                        if len(feat_title) < self.title_len:
                            feat_title.extend([0] * (self.title_len - len(feat_title)))
                        self.title_list.append(feat_title[:self.title_len])

                        ''' article '''
                        feat_article = []
                        words_article = []
                        wordsplit = jieba.cut(article, cut_all=False, HMM=False)
                        for qseg in wordsplit:
                            qseg = re.sub(r'\\s+', ' ', qseg).strip()
                            if qseg:
                                try:
                                    feat_article.append(self.w2index[qseg])
                                    words_article.append(qseg)
                                except KeyError as e:
                                    continue
                        if len(feat_article) < self.article_len:
                            feat_article.extend([0] * (self.article_len - len(feat_article)))
                        self.article_list.append(feat_article[:self.article_len])

                        ''' label '''
                        label = int(id)
                        self.Y_list.append(label)
                        idx += 1
                        if(idx%20000 == 1):
                            print (feat_title, feat_article[:self.article_len])
                            #print (line, words, words_article)

    def get_train_test(self):
        train_title, test_title, train_article, test_article, train_label, test_label \
        = train_test_split(self.title_list, self.article_list, self.Y_list, test_size=self.ratio)
        return  [np.array(train_article), np.array(train_title)], np.array(train_label),\
                [np.array(test_article), np.array(test_title)], np.array(test_label)\
                 
    def get_predict(self):
        return np.array(self.title_list)

    def get_unigram(self, s):
        return list(s)

    def get_bigram(self, s):
        bigram, index = [], 0
        while index < len(s) - 1:
            bigram.append(s[index: index + 2])
            index += 1
        return bigram

    def get_trigram(self, s):
        trigram, index = [], 0
        while index < len(s) - 2:
            trigram.append(s[index: index + 3])
            index += 1
        return trigram

if __name__ == '__main__':
    conf_file = sys.argv[1]
    m = DataLoader()
    m.set_w2v(path=dirPrefix + 'query.skip.vec.win3')
    m.build(conf_file)
    #m.save_dict("vocab.txt")
    train_t, train_label, test_t, test_label = m.get_train_test()

