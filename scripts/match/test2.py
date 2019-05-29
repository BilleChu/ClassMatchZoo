import numpy as NP
from gensim.models import KeyedVectors
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Dropout, Bidirectional, Concatenate, Flatten, Add, ZeroPadding1D, Dot
from keras.layers import LSTM, MaxPooling1D, Embedding, Convolution1D, Average, Activation, Multiply, Lambda
from keras.layers import Conv2D, Reshape
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, LearningRateScheduler, Callback
from keras.utils import plot_model
from keras import backend as K
import codecs

import os
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
import re

TITLE_LENGTH = 30
CONTENT_LENGTH = 500
DEFAULT_WORD_ID = 0
title_hidden_dimension = 100
filter_count = 32
content_hidden_dimension = 100
KERNEL_SIZE = 4

def build_network(cnn_block_count):
    global TITLE_LENGTH, CONTENT_LENGTH, KERNEL_SIZE
    title_length = int(TITLE_LENGTH)
    content_length = int(CONTENT_LENGTH)
    kernel_size = int(KERNEL_SIZE)
    
    word_vector_dimension = 100
    vocabulary_size = 10000
    
    word_embedding_layer = Embedding( vocabulary_size,
                                word_vector_dimension,
                                # input_length=title_length,
                                trainable=False,
                                name='word_embedding_layer' )
    position_embedding_layer = Embedding( input_dim=content_length + 1, output_dim=word_vector_dimension, name='position_embedding_layer' )
    print( 'position embedding shape:(%d,%d)' % (content_length + 1, word_vector_dimension) )

    # Input block ################
    # title
    title_word_sequence_input = Input(shape=(title_length,), dtype='int32', name='title_word_input')
    print( 'title word input shape:' + str(title_word_sequence_input.shape) )
    title_word_embedding_sequences = word_embedding_layer( title_word_sequence_input )
    print( 'title word embedding sequences shape:' + str(title_word_embedding_sequences.shape) )
    
    title_position_sequence_input = Input( shape=(title_length,), dtype='int32', name='title_position_input' )
    print( 'title position input shape:' + str( title_position_sequence_input.shape ) )
    title_position_embedding_sequence = position_embedding_layer( title_position_sequence_input )
    print( 'title position embedding sequences shape:' + str( title_position_embedding_sequence.shape ) )
    
    title_word_position_embedding_sequences = Add( name='title_word_plus_position_layer' )( [title_word_embedding_sequences, title_position_embedding_sequence] )
    
    # content
    content_word_sequence_input = Input(shape=(content_length,), dtype='int32', name='content_word_input')
    print( 'content word input shape:' + str(content_word_sequence_input.shape) )
    content_word_embedded_sequences = word_embedding_layer(content_word_sequence_input)
    
    content_position_sequence_input = Input(shape=(content_length,), dtype='int32', name='content_position_input')
    print( 'content position input shape:' + str(content_position_sequence_input.shape) )
    content_position_embedding_sequences = position_embedding_layer( content_position_sequence_input )
    print( 'content position embedding sequences shape:' + str(content_position_embedding_sequences.shape) )
    
    content_word_position_embedding_sequences = Add( name='content_word_plus_position_layer' )( [content_word_embedded_sequences, content_position_embedding_sequences] )
    print( 'content word position embedding sequence shape:' + str(content_word_position_embedding_sequences) )
    
    # convoluational blocks ####################
    # CNV blocks
    filter_size = 32
    kernel_size = (KERNEL_SIZE, 20)
    block_size = 1

    x_layer = Reshape((content_length, word_vector_dimension, 1))(content_word_position_embedding_sequences)
    x_layer = Conv2D(filter_size, kernel_size, activation='linear', padding='same')(x_layer)
    res_layer = x_layer
    print( 'conv filter size %d, block count %d' % (filter_size, cnn_block_count) )
    
    for i in range(0, cnn_block_count):
        if (i == cnn_block_count-1):
            filter_size = 1
        conv_layer_a = Conv2D( filter_size, kernel_size, activation='linear', padding='same',name='conv_block_conv_a_%d' % i )( x_layer )
        conv_layer_b = Conv2D( filter_size, kernel_size, activation='sigmoid', padding='same',name='conv_block_conv_b_%d' % i )( x_layer )
        x_layer = Multiply( name='glu_layer_%d' % i )( [conv_layer_a, conv_layer_b] )
        if (i%block_size == 0 and i != cnn_block_count-1):
            x_layer = Add( name='residual_layer_%d' % i )([ x_layer, res_layer ])
            res_layer = x_layer
        print( 'residual layer shape:' + str(x_layer.shape) )
    
    x_layer = Reshape((content_length, word_vector_dimension))(x_layer)
    content_vector_raw_layer = Bidirectional(LSTM(word_vector_dimension, dropout_W=0.2, dropout_U=0.2), merge_mode='concat', name='content_vector_original_output_layer')(x_layer)
    content_vector_output_layer = Dense( word_vector_dimension * 2, activation='tanh', name='content_vector_output_layer' )( content_vector_raw_layer ) #通过一个矩阵将title & content投影到相同空间
    
    # title #############
    title_words_lstm = LSTM(title_hidden_dimension, return_sequences=True, dropout_W=0.2, dropout_U=0.2, name='title_words_lstm')( title_word_position_embedding_sequences )
    title_semantic_layer = Bidirectional(LSTM(title_hidden_dimension, dropout_W=0.2, dropout_U=0.2), merge_mode='concat', name='title_semantic_layer')(title_words_lstm)
    print( 'title semantic shape: %s' % str(title_semantic_layer.shape))
    title_attention_layer = Dense( title_hidden_dimension * 2, activation='softmax', name='title_attention_layer' )( title_semantic_layer )
    title_vector_output_layer = Multiply(name='title_vector_original_output_layer')( [title_semantic_layer, title_attention_layer] )
    
    # similarity class
    print( 'title vector output shape:%s, content vector output shape:%s' % (str(title_vector_output_layer.shape), str(content_vector_output_layer.shape)) )
    #x = Subtract( name='title-content-diff' )( [title_vector_output_layer, content_vector_output_layer] )
    similarity_output_layer = Dot( axes=1, normalize=True, name='similarity_output' )( [title_vector_output_layer, content_vector_output_layer] )
    print( 'similarity_output_layer shape: %s' % str(similarity_output_layer.shape) )
    #similarity_output_layer = Dense( 1, activation='sigmoid', name='similarity_output' )( x )
    
    fc_1_size = int(content_vector_output_layer.shape[ 1 ] * 2)
    fc_2_size = int(content_vector_output_layer.shape[ 1 ] * 2)
    # domain class
    domain_x = Dense( fc_1_size, activation='elu', name='domain_fc-1-rnn_cnn' )( content_vector_output_layer )
    domain_x = Dense( fc_2_size, activation='elu', name='domain_fc-2-rnn_cnn' )( domain_x )
    domain_x = Dropout(0.5)(domain_x)
    domain_output_layer = Dense( 10, activation='softmax', name='domain_output_layer' )( domain_x )
    
    # news class
    news_x = Dense( fc_1_size, activation='elu', name='news_fc-1-rnn_cnn' )( content_vector_output_layer )
    news_x = Dense( fc_2_size, activation='elu', name='news_fc-2-rnn_cnn' )( news_x )
    news_x = Dropout(0.5)(domain_x)
    news_output_layer = Dense( 10, activation='softmax', name='news_output_layer' )( news_x )
    
    model = Model(inputs=[title_word_sequence_input, title_position_sequence_input, content_word_sequence_input, content_position_sequence_input], outputs=[similarity_output_layer, domain_output_layer, news_output_layer])
    model.compile(loss={ 'similarity_output':'binary_crossentropy', 'domain_output_layer':'categorical_crossentropy', 'news_output_layer':'categorical_crossentropy'}, loss_weights={'similarity_output':1.0, 'domain_output_layer':0.8, 'news_output_layer':0.8}, optimizer='adam', metrics=['acc'])
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    plot_model(model, show_shapes=True)

build_network(4)