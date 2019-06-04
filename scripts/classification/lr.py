#! /usr/bin python
import sys
import tensorflow as tf
sys.path.append("..")
from tensorflow.keras.layers import Embedding, Dense, Input, Lambda, Activation, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from basic_model import BasicModel
from module.static_history import StaticHistory, Checkpoint
import tensorflow.keras.backend as K

class Lr(BasicModel):

    def __init__(self, conf):
        super(Lr, self).__init__(conf)
        self.name = "Lr"
        self.set_param_list(["batch_size",
                              "embedding_size",
                              "vocab_size",
                              "hidden_layer",
                              "classes",
                              "sequence_len"])
        self.set_conf(conf)
        if not self.check():
            raise TypeError("conf is not complete")
        print ("init completed")

    def set_conf(self, conf):
        if not isinstance(conf, dict):
            raise TypeError("conf should be a dict")

        self.set_default("batch_size", 32)
        self.set_default("embedding_size", 1)
        self.set_default("vocab_size", 2000000)
        self.set_default("hidden_layer", 1)
        self.set_default("classes", 2)
        self.set_default("sequence_len", 15)
        self.param_val.update(conf)

    def mean(self, x):
        return K.mean(x, axis=1)

    def build(self):
        input       = Input((None,))
        embed       = Embedding(input_dim=self.get_param("vocab_size"), output_dim=1)
        inputEmbed  = embed(input)
        allsum      = Lambda(self.mean)(inputEmbed)
        out_        = Dense(self.get_param("classes"), activation="sigmoid")(allsum)
        self.model  = Model(inputs=input, outputs=out_)

        adam        = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
        self.model.summary()

    def train(self, train_data, train_label, test_data, test_label):
        history = StaticHistory(test_data, test_label, self.categories)
        checkpoint  = Checkpoint()
        self.model.fit(train_data, train_label, batch_size=self.get_param("batch_size"),\
                       epochs=self.get_param("epochs"), callbacks = [history], \
                       verbose=1, validation_data=[test_data, test_label])
    
if __name__ == '__main__':
    conf = {}
    a = Lr(conf)
    a.build()











