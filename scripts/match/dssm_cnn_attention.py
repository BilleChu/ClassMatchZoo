#! /usr/bin/python
#coding:utf-8
import sys
import numpy as np
sys.path.append("..")

from keras.layers import Input, MaxPool1D, Permute, Embedding, Dense, Activation, Flatten, Convolution1D,\
                         Dropout, Multiply, Concatenate, BatchNormalization, Dot, Reshape, Lambda, Add

from module.static_history import Checkpoint
from keras.models import Model
import keras.backend as K
import tensorflow as tf
from basic_model import BasicModel
from keras.utils import plot_model

class DSSM_CNN_ATTENTION(BasicModel):
    def __init__(self, conf):
        super(DSSM_CNN_ATTENTION, self).__init__(conf)
        print("Initalizing...")
        self.name = "DSSM_CNN_ATTENTION"
        self.set_conf(conf)
        if not self.check():
            raise TypeError("conf is not complete")
        print ("init completed")
        self.set_default("title_filter_num", 128)
        self.set_default("title_filter_size", [3, 4, 5])
        self.set_default("title_block_size", 2)
        self.set_default("article_filter_num", 128)
        self.set_default("article_filter_size", [16, 16, 32, 32])
        self.set_default("article_block_size", 2)
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

    def glu(self, x_layer, name):
        x_layer = Convolution1D(self.get_param(name+"_filter_num"), self.get_param(name+"_filter_size")[0], activation='linear', padding='same')( x_layer )
        res_layer = x_layer
        for i in range(len(self.get_param(name+"_filter_size"))):
            conv_layer_a = Convolution1D( self.get_param(name+"_filter_num"), self.get_param(name+"_filter_size")[i], activation='linear', padding='same')( x_layer )
            conv_layer_b = Convolution1D( self.get_param(name+"_filter_num"), self.get_param(name+"_filter_size")[i], activation='sigmoid', padding='same')( x_layer )
            x_layer = Multiply()( [conv_layer_a, conv_layer_b] )
            conv_block_output_layer = Dense(self.get_param(name+"_features_dim"), activation='elu')( x_layer )  
            if (i%self.get_param(name + "_block_size") == 0):
                x_layer = Add()([ x_layer, res_layer ])
                res_layer = x_layer
        x_layer = Convolution1D(self.get_param(name+"_features_dim"), self.get_param(name+"_filter_size")[0], activation='linear', padding='same')( x_layer )
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


        article_tower = self.glu(embedded_article, "article")

        ''' title model '''
        title_input = Input(shape=(self.title_max_length,), dtype='int32')
        embedded_title = embedder_article(title_input)
        title_tower = self.glu(embedded_title, "title")
        print (title_tower.shape)
        title_tower = Permute((2, 1))(title_tower)
        def matmul(x):
            return K.batch_dot(x[0], x[1])
        output = Lambda(matmul)([article_tower, title_tower])
        print (output.shape)
        output = MaxPool1D(pool_size=self.article_max_length, padding='valid')(output)
        output = Reshape((self.title_max_length,))(output)
        output = Dense(1)(output)
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
            "article_max_length": 20,
            "title_max_length": 250,
            "article_hidden_dims": 100,
            "title_hidden_dims": 100,
            "epochs": 100}
    binaryClf = DSSM_CNN_ATTENTION(conf)
    binaryClf.build()
