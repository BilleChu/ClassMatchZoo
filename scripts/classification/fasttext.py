#! /usr/bin python
import tensorflow as tf
import sys
sys.path.append("..")
from keras.layers import Embedding, Dense, Input, Lambda, Activation
from keras.models import Model
from keras.activations import sigmoid
from keras.models import load_model
from keras.optimizers import Adam
from basic_model import BasicModel
from module.static_history import StaticHistory, Checkpoint
import keras.backend as K

class Fasttext(BasicModel):
    def __init__(self, conf):
        super(Fasttext, self).__init__(conf)
        self.name = "fasttext"
        self.set_param_list(["batch_size",
                              "embedding_size",
                              "vocab_size",
                              "hidden_layer",
                              "classes"])
        self.set_conf(conf)
        if not self.check():
            raise TypeError("conf is not complete")
        print ("init completed")

    def set_conf(self, conf):
        if not isinstance(conf, dict):
            raise TypeError("conf should be a dict")

        self.set_default("batch_size", 32)
        self.set_default("embedding_size", 128)
        self.set_default("vocab_size", 200000)
        self.set_default("hidden_layer", 128)
        self.set_default("classes", 2)
        self.param_val.update(conf)

    def build(self):
        input = Input((self.get_param("sequence_len"),))
        embed = Embedding(input_dim=self.get_param("vocab_size") ,
                          output_dim=self.get_param("embedding_size"),
                          weights=[self.weights], trainable=False)
        inputEmbed = embed(input)

        def mean(self, x):
            return K.mean(x, axis=1)

        allsum = Lambda(self.mean)(inputEmbed)
        x = Activation('sigmoid')(allsum)
        out_ = Dense(self.get_param("classes"), activation='softmax')(x)
        self.model = Model(inputs=input, outputs=out_)

        adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
        self.model.summary()

    def train(self, train_data, train_label, test_data, test_label):
        history = StaticHistory(test_data, test_label, self.categories)
        checkpoint  = Checkpoint()
        self.model.fit(train_data, train_label, batch_size=self.get_param("batch_size"),\
                       epochs=self.get_param("epochs"), callbacks=[history], \
                       verbose=1, validation_data=[test_data, test_label])

if __name__ == '__main__':
    conf = {}
    a = Fasttext(conf)
    a.build()











