#! /usr/bin/python
#coding:utf-8
import sys
sys.path.append("..")
import tensorflow as tf
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Embedding, Dense, Activation,\
                         Dropout, Multiply, Concatenate, BatchNormalization

from tensorflow.keras.models import Model
from basic_model import BasicModel
from module.static_history import StaticHistory, Checkpoint
import tensorflow.keras.backend as K
K.set_learning_phase(1)

class TextRnn(BasicModel):
    def __init__(self, conf):
        super(TextRnn, self).__init__(conf)
        print("Initalizing...")
        self.name = "TextRnn"
        self.set_conf(conf)
        if not self.check():
            raise TypeError("conf is not complete")
        print ("init completed")

    def set_conf(self, conf):
        if not isinstance(conf, dict):
            raise TypeError("conf should be a dict")
        self.set_default("batch_size", 32)
        self.set_default("embedding_size", 128)
        self.set_default("vocab_size", 2000000)
        self.set_default("classes", 2)
        self.set_default("sequence_len", 15)
        self.set_default("epochs", 2)
        self.set_default("hidden_dims", 128)
        self.param_val.update(conf)

    def build(self):
        print("Start to build the TextRNN model")
        input_ = Input((self.get_param("sequence_len"),))
        embed  = Embedding(input_dim=self.get_param("vocab_size") ,
                           output_dim=self.get_param("embedding_size"),
                           weights=[self.weights], trainable=False)(input_)

        x = Bidirectional(LSTM(units=self.get_param("hidden_dims"),
                               dropout_W=0.2,
                               dropout_U=0.2,
                               return_sequences=False),
                               merge_mode='concat')(embed)

        x = Dense(self.get_param("hidden_dims") * 2, activation="sigmoid")(x)
#        x = BatchNormalization()(x)
#        x = Dropout(0.5)(x)
        output = Dense(self.get_param("classes"), activation='softmax', name='output')(x)
        self.model = Model(inputs=input_, outputs=output)
        self.model.compile(optimizer="adam",
                           loss="categorical_crossentropy",
                           metrics=["accuracy"])
        self.model.summary()
        print("Get the model build work Done!")

    def train(self, train_data, train_label, test_data, test_label):
        history     = StaticHistory(test_data, test_label, self.categories)
        checkpoint  = Checkpoint()
        self.model.fit(train_data, train_label, batch_size=self.get_param("batch_size"),\
                       epochs=self.get_param("epochs"), callbacks = [history], \
                       verbose=1, validation_data=[test_data, test_label])

if __name__ == "__main__":

    binaryClf = TextRnn({})
    binaryClf.build()
