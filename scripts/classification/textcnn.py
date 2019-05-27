#! /usr/bin python
from __future__ import print_function
import tensorflow as tf 
import keras
import sys
sys.path.append("..")

from keras.layers import Embedding, Dense, Input, Dropout, Concatenate, \
                         Activation, Conv2D, Reshape, MaxPool2D, Flatten

from keras.models import Model
from keras.optimizers import Adam
from basic_model import BasicModel
from module.static_history import StaticHistory, Checkpoint, LrateScheduler
import keras.backend as K
K.set_learning_phase(1)

class TextCnn(BasicModel):

    def __init__(self, conf):
        super(TextCnn, self).__init__(conf)
        self.name = "TextCnn"
        self.set_param_list(["batch_size",
                              "embedding_size",
                              "vocab_size",
                              "classes",
                              "sequence_len"])
        self.set_conf(conf)
        if not self.check():
            raise TypeError("conf is not complete")
        print ("init completed", end="\n")

    def set_conf(self, conf):
        if not isinstance(conf, dict):
            raise TypeError("conf should be a dict")

        self.set_default("batch_size", 32)
        self.set_default("embedding_size", 128)
        self.set_default("vocab_size", 2000000)
        self.set_default("filter_size", [3, 4, 5])
        self.set_default("classes", 2)
        self.set_default("sequence_len", 15)
        self.set_default("filter_num", 32)
        self.set_default("epochs", 2)
        self.param_val.update(conf)

    def build(self):
        input_ = Input((self.get_param("sequence_len"),))
        embed  = Embedding(input_dim=self.get_param("vocab_size") ,
                           output_dim=self.get_param("embedding_size"),
                           weights=[self.weights], trainable=False)(input_)
        embed  = Reshape((self.get_param("sequence_len"), self.get_param("embedding_size"), 1))(embed)
        channels = []
        for i in range(len(self.get_param("filter_size"))):
            conv  = Conv2D(self.get_param("filter_num"), kernel_size=(self.get_param("filter_size")[i], self.get_param("embedding_size")))(embed)
            maxpool = MaxPool2D(pool_size=(self.get_param("sequence_len") - self.get_param("filter_size")[i] + 1, 1), strides=(1, 1), padding='valid')(conv)
            channels.append(maxpool)

        concat  = Concatenate(axis=1)(channels)
        flat    = Flatten()(concat)
        dropout = Dropout(.5)(flat)
        output_ = Dense(units=self.get_param("classes"), activation='softmax')(dropout)
        self.model = Model(inputs=input_, outputs=output_)
        adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
        self.model.summary()

    def train(self, train_data, train_label, test_data, test_label):
        checkpoint  = Checkpoint()
        scheduler   = LrateScheduler()
        history     = StaticHistory(train_data, train_label, self.categories)
        self.model.fit(train_data, train_label, batch_size=self.get_param("batch_size"),\
                       epochs=self.get_param("epochs"), callbacks=[history, scheduler],\
                       verbose=1, validation_data=[test_data, test_label])

if __name__ == '__main__':
    conf = {}
    a = TextCnn(conf)
    a.build()











