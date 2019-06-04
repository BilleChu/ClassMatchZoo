#! /usr/bin/python
#coding:utf-8
import sys
import numpy as np
sys.path.append("..")

from keras.layers import Input, MaxPool2D, Conv2D, Embedding, Dense, Activation, Flatten, \
                         Dropout, Multiply, Concatenate, BatchNormalization, Reshape, Lambda, Permute

from module.static_history import Checkpoint
from keras.models import Model
import keras.backend as K
import tensorflow as tf
from basic_model import BasicModel
from keras.utils import plot_model

class Match_pyramid(BasicModel):
    def __init__(self, conf):
        super(Match_pyramid, self).__init__(conf)
        print("Initalizing...")
        self.name = "Match_pyramid"
        self.set_conf(conf)
        if not self.check():
            raise TypeError("conf is not complete")
        print ("init completed")
        self.set_default("filter_num", [32, 32, 32])
        self.set_default("kernel_size", [(3, 3), (3, 3), (3, 3)])
        self.set_default("pool_size", [(3, 3), (3, 3), (3, 3)])
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

    def pyramid(self, x_layer):
        for i in range(len(self.get_param("kernel_size"))):
            x_layer = Conv2D(self.get_param("filter_num")[i], self.get_param("kernel_size")[i], activation='elu')( x_layer )
            x_layer = MaxPool2D(self.get_param("pool_size")[i])(x_layer)
        print (x_layer.shape)
        x_layer = Flatten()(x_layer)
        return x_layer

    def build(self):
        print("Start to build the DL model")
        ''' article model '''
        embedder_article = Embedding(input_dim=self.get_param("vocab_size"),
                                      output_dim=self.article_features_dim,
                                      #weights=[self.weights],
                                      trainable=False)

        article_input = Input(shape=(self.article_max_length,), dtype='int32')
        embedded_article = embedder_article(article_input)

        ''' title model '''
        title_input = Input(shape=(self.title_max_length,), dtype='int32')
        embedded_title = embedder_article(title_input)
        embedded_title = Permute((2, 1))(embedded_title)
        #embedded_title = Reshape((self.title_features_dim, self.title_max_length))(embedded_title)
        def matmul(x):
            return K.batch_dot(x[0], x[1])
        interact_layer = Lambda(matmul)([embedded_article, embedded_title])
        interact_layer = Reshape((self.article_max_length, self.title_max_length, 1))(interact_layer)
        interact_layer = self.pyramid(interact_layer)
        output = Dense(1, activation='sigmoid')(interact_layer)
        self.model = Model(inputs=[article_input, title_input], outputs=output)
        self.model.compile(optimizer="adam",
                           loss="mean_squared_error",
                           metrics=["accuracy"])
        self.model.summary()
        plot_model(self.model, show_shapes=1)
        print("Get the model build work Done!")

    def train(self, train_data, train_label, test_data, test_label):
        #[train_article, train_title] = train_data
        #[test_article, test_title]   = test_data
        self.model.fit(train_data, train_label,
                       batch_size=128, epochs=self.get_param("epochs"),
                       validation_data = (test_data, test_label), verbose=1)

if __name__ == "__main__":
    conf = {"title_features_dim": 100,
            "article_features_dim": 100,
            "vocab_size": 1000,
            "article_max_length": 200,
            "title_max_length": 250,
            "article_hidden_dims": 100,
            "title_hidden_dims": 100,
            "epochs": 100}
    binaryClf = Match_pyramid(conf)
    binaryClf.build()
