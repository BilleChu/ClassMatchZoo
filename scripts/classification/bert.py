# coding=utf-8


import tensorflow as tf
import os
import sys

sys.path.append(os.path.abspath(__file__) + "../../bert/")
import tokenization, modeling
# 配置文件
data_root = '/mnt/hgfs/share/BertEmbedding/'
bert_config_file = data_root + 'bert_config.json'
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
init_checkpoint = data_root + 'bert_model.ckpt'
bert_vocab_file = data_root + 'vocab.txt'

# graph
input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')

# 初始化BERT
model = modeling.BertModel(
    config=bert_config,
    is_training=False,
    input_ids=input_ids,
    input_mask=None,
    token_type_ids=None,
    use_one_hot_embeddings=False)

# 加载bert模型
tvars = tf.trainable_variables()
(assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
tf.train.init_from_checkpoint(init_checkpoint, assignment)
# 获取最后一层和倒数第二层。
embedding_output = model.get_embedding_output()
encoder_last_layer = model.get_sequence_output()
encoder_last2_layer = model.all_encoder_layers[-2]
tmp = model.input_ids
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    token = tokenization.CharTokenizer(vocab_file=bert_vocab_file)
    query = u'Jack,请回答1988, UNwant\u00E9d,running'
    split_tokens = token.tokenize(query)
    word_ids = token.convert_tokens_to_ids(split_tokens)
    fd = {input_ids: [word_ids]}
    embed, last, tm = sess.run([embedding_output, encoder_last_layer, tmp], feed_dict=fd)
    pass
