#! /usr/bin/python
#coding:utf-8
import sys
import numpy as np
sys.path.append("..")
from keras.layers import Input, MaxPool2D, Conv2D, Embedding, Dense, Activation, Flatten,\
                         Dropout, Multiply, Concatenate, BatchNormalization, Dot, Reshape

from module.static_history import Checkpoint
from keras.models import Model
import keras.backend as K
import tensorflow as tf
from basic_model import BasicModel

class DSSM_CNN_ATTENTION(BasicModel):
    def __init__(self, conf):
        super(DSSM_CNN, self).__init__(conf)
        print("Initalizing...")
        self.name = "DSSM_CNN_ATTENTION"
        self.set_conf(conf)
        if not self.check():
            raise TypeError("conf is not complete")
        print ("init completed", end="\n")
        self.set_default("filter_num", 32)
        self.set_default("filter_size", [3, 4, 5])
        self.title_features_dim = self.get_param("title_features_dim")
        self.article_features_dim = self.get_param("article_features_dim")
        self.article_max_length = self.get_param("article_max_length")
        self.title_max_length = self.get_param("title_max_length")
        self.article_hidden_dims = self.get_param("article_hidden_dims")
        self.title_hidden_dims = self.get_param("title_hidden_dims")

    def set_conf(self, conf):
        if not isinstance(conf, dict):
            raise TypeError("conf should be a dict")
        self.param_val.update(conf)

    def tower(self, embed, name):
        embed  = Reshape((self.get_param(name + "_max_length"), self.get_param(name + "_features_dim"), 1))(embed)
        channels = []
        for i in range(len(self.get_param("filter_size"))):
            conv    = Conv2D(self.get_param("filter_num"), kernel_size=(self.get_param("filter_size")[i], self.get_param(name + "_features_dim")))(embed)
            maxpool = MaxPool2D(pool_size=(self.get_param(name + "_max_length") - self.get_param("filter_size")[i] + 1, 1), strides=(1, 1), padding='valid')(conv)
            channels.append(maxpool)
        concat  = Concatenate(axis=1)(channels)
        flat    = Flatten()(concat)
        return flat

    def build(self):
        print("Start to build the DL model")
        ''' article model '''
        embedder_article = Embedding(input_dim=self.get_param("vocab_size"),
                                      output_dim=self.article_features_dim,
                                      weights=[self.weights],
                                      trainable=False)

        article_input = Input(shape=(self.article_max_length,), dtype='int32')
        embedded_article = embedder_article(article_input)
        article_tower = self.tower(embedded_article, "article")

        ''' title model '''
        title_input = Input(shape=(self.title_max_length,), dtype='int32')
        embedded_title = embedder_article(title_input)
        title_tower = self.tower(embedded_title, "title")

        output = Dot(axes=-1)([article_tower, title_tower])
        output = Activation(activation='sigmoid')(output)

        #output = Concatenate()([article_tower, title_tower])
        #output = Dense(1, activation='sigmoid')(output)

        self.model = Model(inputs=[article_input, title_input], outputs=output)
        self.model.compile(optimizer="adam",
                           loss="mean_squared_error",
                           metrics=["accuracy"])
        self.model.summary()
        print("Get the model build work Done!")

    def train(self, train_data, train_label, test_data, test_label):
        #[train_article, train_title] = train_data
        #[test_article, test_title]   = test_data
        self.model.fit(train_data, train_label,
                       batch_size=128, epochs=self.get_param("epochs"),
                       validation_data = (test_data, test_label), verbose=1)

if __name__ == "__main__":

    binaryClf = DSSM_CNN({})
    binaryClf.build()
