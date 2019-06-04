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
        self.title_len = 5
        self.article_len = 10

    def set_params(self, params):
        super(DataLoader, self).set_params(params)
        if ("title_len" in params):
            self.title_len = params["title_len"]
        else:
            print ("title_len default: ", self.title_len)
        if ("article_len" in params):
            self.article_len = params["article_len"]
        else:
            print ("article_len default: ", self.article_len)

    def build(self, conf_file):
        with codecs.open(conf_file, "r", encoding='utf8') as files:
            self.title_list = []
            self.Y_list = []
            self.article_list = []
            idx = 0
            for line in files:
                id, filename = line.strip().split()
                print(id, filename)
                counter = 0
                with codecs.open(filename, 'r', encoding='utf8') as sample_file:
                    print(filename,"open success!!!")
                    for line in sample_file:
                        counter += 1
                        try:
                            title, article = line.split("\t")
                        except Exception as e:
                            print ("error", counter)
                            continue

                        feat_title = []
                        words = []
                        for func in self.gram_func:
                            feat_title, words = func(title, feat_title, words)
                            
                        ''' title '''

                        if len(feat_title) < self.title_len:
                            feat_title.extend([0] * (self.title_len - len(feat_title)))
                        self.title_list.append(feat_title[:self.title_len])

                        ''' article '''
                        feat_article = []
                        words_article = []
                        for func in self.gram_func:
                            feat_article, words_article = func(article, feat_article, words_article)

                        if len(feat_article) < self.article_len:
                            feat_article.extend([0] * (self.article_len - len(feat_article)))
                        self.article_list.append(feat_article[:self.article_len])

                        ''' label '''
                        label = int(id)
                        self.Y_list.append(label)
                        idx += 1
                        if(idx%20000 == 1):
                            print (feat_title, feat_article)
                            #print (line, words, words_article)

    def get_train_test(self):
        train_title, test_title, train_article, test_article, train_label, test_label \
        = train_test_split(self.title_list, self.article_list, self.Y_list, test_size=self.ratio)
        return  [np.array(train_article), np.array(train_title)], np.array(train_label),\
                [np.array(test_article), np.array(test_title)], np.array(test_label)\
                 
if __name__ == '__main__':
    conf_file = sys.argv[1]
    m = DataLoader()
    m.set_w2v(path="/mnt/hgfs/share/pornCensor/query.skip.vec.win3")
    m.build(conf_file)


