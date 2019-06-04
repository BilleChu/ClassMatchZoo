#! /usr/bin/python
#coding:utf-8

import numpy as np
import sys
import random
sys.path.append("..")

from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import  Input, Bidirectional, LSTM, Embedding, Dense, Activation, \
                          BatchNormalization, Dropout, Multiply, Concatenate
from tensorflow.keras.models import Model
from module.attention import Attention
from basic_model import BasicModel
from module.static_history import StaticHistory, Checkpoint, LrateScheduler
import tensorflow.keras.backend as K
import tensorflow as tf

class binaryClassifier(BasicModel):
    def __init__(self, conf):
        super(binaryClassifier, self).__init__(conf)
        print("Initalizing...")
        self.name = "TextCnn"
        self.set_conf(conf)
        if not self.check():
            raise TypeError("conf is not complete")
        print ("init completed")
        print(self.param_val)
        self.word_features_dim  = self.get_param("word_features_dim")
        self.char_features_dim  = self.get_param("char_features_dim")
        self.char_max_length    = self.get_param("char_max_length")
        self.word_max_length    = self.get_param("word_max_length")
        self.char_hidden_dims   = self.get_param("char_hidden_dims")
        self.word_hidden_dims   = self.get_param("word_hidden_dims")

    def set_conf(self, conf):
        if not isinstance(conf, dict):
            raise TypeError("conf should be a dict")
        self.param_val.update(conf)

    def set_embedding(self, w_kv, c_kv):
        self.char_embedding = c_kv
        self.word_embedding = w_kv

    def build(self):
        print("Start to build the DL model")
        ''' chars model '''
        embedder_chars  = Embedding(input_dim=len(self.char_embedding),
                                    output_dim=self.char_features_dim,
                                    weights=[self.char_embedding],
                                    trainable=True)
        char_input      = Input(shape=(self.char_max_length,), dtype='int32')
        embedded_chars  = embedder_chars(char_input)

        x = Bidirectional(LSTM(self.char_hidden_dims,
                               dropout_W=0.2,
                               dropout_U=0.2,
                               return_sequences=True),
                               merge_mode='concat')(embedded_chars)
        #x = Attention(self.char_hidden_dims * 2, x)  # biDirection
        x = Attention(self.char_hidden_dims * 2)(x)  # biDirection
        x = Dense(self.char_hidden_dims, activation="sigmoid")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        ''' word model '''
        embedder_words  = Embedding(input_dim=len(self.word_embedding),
                                    output_dim=self.word_features_dim,
                                    weights=[self.word_embedding],
                                    trainable=True)
        word_input      = Input(shape=(self.word_max_length,), dtype='int32')
        embedded_words  = embedder_words(word_input)

        y = Bidirectional(LSTM(self.word_hidden_dims,
                               dropout_W=0.2,
                               dropout_U=0.2,
                               return_sequences=True),
                               merge_mode='concat')(embedded_words)

        #y = Attention(self.word_hidden_dims * 2, y)  # biDirection
        y = Attention(self.word_hidden_dims * 2)(y)  # biDirection
        y = Dense(self.word_hidden_dims, activation="sigmoid")(y)
        y = BatchNormalization()(y)
        y = Dropout(0.5)(y)
        
        merged = Concatenate()([x, y])
        output = Dense(self.get_param("classes"), activation='softmax', name='binary')(merged)
        self.model = Model(inputs=[char_input, word_input], outputs=output)
        self.model.compile(optimizer="adam",
                           loss="categorical_crossentropy",
                           metrics=["accuracy"])
        self.model.summary()
        print("Get the model build work Done!")

    def train(self, train_data, train_label, test_data, test_label):
        #[train_chars, train_words] = train_data
        #[test_chars, test_words]   = test_data
        static_history  = StaticHistory(test_data, test_label, self.categories)
        learingrate     = LrateScheduler()
        checkpoint      = Checkpoint()
        self.model.fit(train_data,
                       train_label,
                       batch_size=128,
                       epochs=self.get_param("epochs"),
                       validation_data = (test_data, test_label),
                       verbose=1)


if __name__ == "__main__":

    binaryClf = binaryClassifier({})
    binaryClf.build()
