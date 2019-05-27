from keras.layers import TimeDistributed, Flatten, RepeatVector, Permute, Lambda,\
                         Multiply, Concatenate, Layer, Dense, Activation
import keras.backend as K
'''
def Attention(hidden_dims, x):
    attention = TimeDistributed(Dense(1, activation='tanh'))(x)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(hidden_dims)(attention)
    attention = Permute((2, 1))(attention)
    x = Multiply()([x, attention])
    x = Lambda(lambda xx: K.sum(xx, axis=1))(x)
    return x

'''                         
class Attention(Layer):
    def __init__(self, hidden_dims, **kwargs):
        self.hidden_dims = hidden_dims
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_layer = Dense(1, activation='tanh')
        super(Attention, self).build(input_shape)

    def call(self, x):
        attention = TimeDistributed(self.dense_layer)(x)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(self.hidden_dims)(attention)
        attention = Permute((2, 1))(attention)
        x = Multiply()([x, attention])
        x = Lambda(lambda xx: K.sum(xx, axis=1))(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.hidden_dims)


