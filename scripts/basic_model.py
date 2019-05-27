#! /usr/bin python
#! coding=utf-8
from __future__ import print_function
from __future__ import absolute_import
from keras.models import load_model

class BasicModel(object):
    def __init__(self, conf):
        self.param_list = []
        self.param_val = conf

    def set_default(self, param, val):
        if (param not in self.param_val):
            self.param_val[param] = val
        else:
            print(param + " is already set to " + str(self.param_val[param]), end="\n")

    def set_conf(self, conf):
        pass

    def set_embedding(self, weights):
        print (weights.shape)
        self.weights = weights

    def load_model(self, path):
        self.model = load_model(path)

    def predict(self, pred_data):
        return self.model.predict(pred_data, batch_size=self.get_param("batch_size"))

    def set_categories(self, categories):
        if isinstance(categories, list):
            self.categories = categories
        else:
            raise TypeError("categories should be a list")

    def get_categories(self):
        return self.categories

    def get_param(self, param):
        if (param in self.param_val):
            return self.param_val[param]
        else:
            raise ValueError("no such param")

    def check(self):
        print (self.param_list)
        for param in self.param_list:
            if param not in self.param_val:
                print ("Error {} is not ready".format(param), end='\n')
                return False
        return True

    def build(self):
        pass

    def set_param_list(self, param_list):
        self.param_list = param_list
