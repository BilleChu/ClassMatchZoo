#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import sys
import codecs
import numpy as np
sys.path.append("..")
from .basic_loader import *
from sklearn.model_selection import train_test_split

class DataLoader(BasicLoader):
    def __init__(self):
        super(DataLoader, self).__init__()
        self.classes = 0
        self.max_len = 15

    def set_params(self, params):
        super(DataLoader, self).set_params(params)
        if ("max_len" in params):
            self.max_len = params["max_len"]
        else:
            print ("max_length default: ", self.max_len)

    def get_categories(self):
        return self.categories

    def build(self, conf_file):
        with codecs.open(conf_file, "r", encoding='utf8') as fi:
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

                with codecs.open(filename, 'r', encoding='utf8') as sample_file:
                    print(filename, "open success!!!")
                    for line in sample_file:
                        feat = []
                        words = []
                        for func in self.gram_func:
                            feat, words = func(line, feat, words)
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

if __name__ == '__main__':
    conf_file = sys.argv[1]
    m = DataLoader()
    m.set_w2v(path="/mnt/hgfs/share/pornCensor/query.skip.vec.win3")
    m.build(conf_file)
    #m.save_dict("vocab.txt")