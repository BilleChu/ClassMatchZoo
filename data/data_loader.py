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
        self.word_vec_len = 100
        self.ratio = 0.2
        self.max_len = 15
        self.classes = 0
        
    def set_params(self, params):
        if ("ratio" in params):
            self.ratio = params["ratio"]
        else:
            print ("train/test ratio default: ", self.ratio)
        if ("max_len" in params):
            self.max_len = params["max_len"]
        else:
            print ("max_length default: ", self.max_len)
        if ("embedding_size" in params):
            self.word_vec_len = params["embedding_size"]
        else:
            print ("embedding_size default: ", self.word_vec_len)

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

    def get_categories(self):
        return self.categories

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
        with open(conf_file, "r") as fi:
            files = fi.readlines()
            self.X_list = []
            self.Y_list = []
            self.classes = len(files)
            self.categories = []
            idx = 0
            classid = 0
            for line in files:
                vs = line.strip().split()
                classname = vs[0]
                self.categories.append(classname)
                filename = vs[1]
                print(classname, filename)
                counter = 0
                with open(filename, 'r') as sample_file:
                    print(filename, "open success!!!")
                    for line in sample_file:
                        counter += 1
                        feat = []
                        words = []
                        try:
                            line = line.strip().lower()
                        except Exception as e:
                            continue
                        wordsplit = jieba.cut(line, cut_all=False, HMM=False)
                        for qseg in wordsplit:
                            qseg = re.sub(r'\\s+', ' ', qseg).strip()
                            if qseg:
                                try:
                                    feat.append(self.w2index[qseg])
                                    words.append(qseg)
                                except KeyError as e:
                                    continue

                        grams = self.get_trigram(line.replace(' ', ''))
                        for gram in grams:
                            if gram in words:
                                continue
                            try:
                                feat.append(self.w2index[gram])
                                words.append(gram)
                            except KeyError as e:
                                continue

                        grams = self.get_bigram(line.replace(' ', ''))
                        for gram in grams:
                            if gram in words:
                                continue
                            try:
                                feat.append(self.w2index[gram])
                                words.append(gram)
                            except KeyError as e:
                                continue

                        grams = self.get_unigram(line.replace(' ', ''))
                        for gram in grams:
                            if gram in words:
                                continue
                            try:
                                feat.append(self.w2index[gram])
                                words.append(gram)
                            except KeyError as e:
                                continue

                        if len(feat) < self.max_len:
                            feat.extend([0] * (self.max_len - len(feat)))
                        self.X_list.append(feat[:self.max_len])
                        label = [0.0] * self.classes # 2 classes
                        label[int(classid)] = 1.0
                        self.Y_list.append(label)
                        idx += 1
                        if(idx%20000 == 1):
                            print (feat)
                            print (line, words)
                classid += 1

    def get_train_test(self):
        print ("Class Num : ", self.classes)
        train_data, test_data, train_label, test_label = train_test_split(self.X_list, self.Y_list, test_size=self.ratio)
        return np.array(train_data), np.array(test_data), np.array(train_label), np.array(test_label)

    def get_predict(self):
        return np.array(self.X_list)

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
    m.save_dict("vocab.txt")
    train_data, test_data, train_label, test_label = m.get_train_test()
    print (test_data[:2], "  ",train_data[:2])