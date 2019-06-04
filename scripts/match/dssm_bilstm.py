#! /usr/bin/python
#coding:utf-8
import numpy as np
import sys
sys.path.append("..")

from tensorflow.keras.layers import Input, Bidirectional, LSTM, Embedding, Dense, Activation,\
                         Dropout, Multiply, Concatenate, BatchNormalization, Dot

from tensorflow.keras.models import Model
from basic_model import BasicModel
from module.static_history import Checkpoint
import tensorflow.keras.backend as K
import tensorflow as tf

class DSSM_biLSTM(BasicModel):
    def __init__(self, conf):
        super(DSSM_biLSTM, self).__init__(conf)
        print("Initalizing...")
        self.name = "DSSM_biLSTM"
        self.set_conf(conf)
        if not self.check():
            raise TypeError("conf is not complete")
        print ("init completed")

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

    def build(self):
        print("Start to build the DL model")
        ''' chars model '''
        embedder_article = Embedding(input_dim=self.get_param("vocab_size"),
                                      output_dim=self.article_features_dim,
                                      weights=[self.weights],
                                      trainable=False)

        article_input = Input(shape=(self.article_max_length,), dtype='int32')
        embedded_article = embedder_article(article_input)

        x = Bidirectional(LSTM(self.article_hidden_dims,
                               dropout_W=0.2,
                               dropout_U=0.2,
                               return_sequences=False),
                               merge_mode='concat')(embedded_article)

        x = Dense(self.article_hidden_dims, activation="sigmoid")(x)
        x = BatchNormalization()(x)
#        x = Dropout(0.5)(x)

        ''' word model '''
        '''
        embedder_title = Embedding (input_dim=self.get_param("vocab_size"),
                                    output_dim=self.article_features_dim,
                                    weights=[self.weights],
                                    trainable=False)
        '''
        title_input = Input(shape=(self.title_max_length,), dtype='int32')
        embedded_title = embedder_article(title_input)

        y = Bidirectional(LSTM(self.title_hidden_dims,
                               dropout_W=0.2,
                               dropout_U=0.2,
                               return_sequences=False),
                               merge_mode='concat')(embedded_title)

        y = Dense(self.title_hidden_dims, activation="sigmoid")(y)
        y = BatchNormalization()(y)
#        y = Dropout(0.5)(y)
        output = Concatenate()([x, y])
        output = Dense(1, activation='sigmoid')(output)

#        output = Dot(axes=-1)([x, y])
#        output = Activation(activation='sigmoid')(output)
        
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

    binaryClf = DSSM_biLSTM({})
    binaryClf.build()
