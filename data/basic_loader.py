#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import jieba
import json
import re
import codecs
import numpy as np
from tool.n_gram import get_bigram, get_unigram, get_trigram

class BasicLoader(object):
    def __init__(self):
        self.w2v = {}
        self.w2index = {}
        self.weights = []
        self.word_vec_len = 100
        self.ratio = 0.2
        self.gram_nums = [0, 3, 2, 1] # 0 means jieba.cut
        self.gram_func = []
        self.inscript_gram_func()
    
    def set_params(self, params):
        if ("ratio" in params):
            self.ratio = params["ratio"]
        else:
            print ("train/test ratio default: ", self.ratio)

        if ("gram_nums" in params):
            self.gram_nums = params["gram_nums"]
        else:
            print ("default gram functions: ", self.gram_func)

        if ("embedding_size" in params):
            self.word_vec_len = params["embedding_size"]
        else:
            print ("embedding_size default: ", self.word_vec_len)
        self.inscript_gram_func()

    def inscript_gram_func(self):
        self.gram_func = []
        for i in self.gram_nums:
            if (0 == i):
                self.gram_func.append(self.add_jieba_tokens)
            elif (1 == i):
                self.gram_func.append(self.add_unigram)
            elif (2 == i):
                self.gram_func.append(self.add_bigram)
            elif (3 == i):
                self.gram_func.append(self.add_trigram)
            else:
                print("No such gram function!")

    def set_w2v(self, path='../../xx'):
        # word2vec gensim style, first line word_num&dims
        with codecs.open(path, "r", encoding='utf8', errors='ignore') as f:
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
        with codecs.open(filepath, "w", encoding='utf8') as fp:
            d = sorted(self.w2index.items(), key=lambda item:item[1])
            for key, value in d:
                fp.write(key+"\n")
#            json.dump(self.w2index, fp)

    def load_dict(self, filepath):
        with codecs.open(filepath, "r", encoding='utf8') as fp:
            index = 0
            for key in fp:
                self.w2index[key] = index
                index += 1
#            self.w2index = json.load(fp)

    def build(self, conf_file):
        pass

    def get_train_test(self):
        pass

    def add_jieba_tokens(self, title, feat_title, words):
        wordsplit = jieba.cut(title, cut_all=False, HMM=False)
        for qseg in wordsplit:
            qseg = re.sub(r'\\s+', ' ', qseg).strip()
            if qseg:
                try:
                    feat_title.append(self.w2index[qseg])
                    words.append(qseg)
                except KeyError as e:
                    continue
        return feat_title, words

    def add_trigram(self, title, feat_title, words):
        grams = get_trigram(title.replace(' ', ''))
        for gram in grams:
            if gram in words:
                continue
            try:
                feat_title.append(self.w2index[gram])
                words.append(gram)
            except KeyError as e:
                continue
        return feat_title, words

    def add_bigram(self,  title, feat_title, words):
        grams = get_bigram(title.replace(' ', ''))
        for gram in grams:
            if gram in words:
                continue
            try:
                feat_title.append(self.w2index[gram])
                words.append(gram)
            except KeyError as e:
                continue
        return feat_title, words

    def add_unigram(self,  title, feat_title, words):
        grams = get_unigram(title.replace(' ', ''))
        for gram in grams:
            if gram in words:
                continue
            try:
                feat_title.append(self.w2index[gram])
                words.append(gram)
            except KeyError as e:
                continue
        return feat_title, words