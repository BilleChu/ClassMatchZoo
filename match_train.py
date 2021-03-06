#!/usr/bin/python
#! -*- coding: utf-8 -*-
import sys
import os
import numpy as np
sys.path.append("./scripts/")
from scripts.match.dssm_bilstm import DSSM_biLSTM
from scripts.match.dssm_textcnn import DSSM_CNN
from scripts.match.dssm_bilstm_attention import DSSM_biLSTM_attention
from scripts.match.match_pyramid import Match_pyramid
from scripts.match.dssm_cnn_attention import DSSM_CNN_ATTENTION
from data.match_data_loader import DataLoader
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_file",
    type=str,
    default="conf/match_conf",
    help="conf_file containes sample files and labels")

parser.add_argument(
    "--w2v_path",
    type=str,
    default="/mnt/hgfs/share/pornCensor/query.skip.vec.win3",
    help="w2v file which provide w2v")

parser.add_argument(
    "--model",
    type=str,
    default="DSSM_CNN",
    help="DSSM_CNN, DSSM_biLSTM, DSSM_biLSTM_attention, Match_pyramid, DSSM_CNN_ATTENTION"
    )
FLAGS, unparsed = parser.parse_known_args()
print ("unparsed: ", unparsed)

params =  { "ratio"         :   0.2,
            "gram_nums"     :   [0], # - mean jieba-cut
            "embedding_size":   100,
            "title_len"     :   24,
            "article_len"   :   512 }
loader = DataLoader()
loader.set_params(params)
loader.set_w2v(FLAGS.w2v_path)
#loader.save_dict("data/title_dict.json")
loader.build(FLAGS.conf_file)
train_data, train_label, test_data, test_label = loader.get_train_test()

conf = { "title_features_dim"   : loader.word_vec_len,
         "article_features_dim" : loader.word_vec_len,
         "vocab_size"           : len(loader.weights),
         "article_max_length"   : loader.article_len,
         "title_max_length"     : loader.title_len,
         "article_hidden_dims"  : 100,
         "title_hidden_dims"    : 100,
         "epochs"               : 100}

if (FLAGS.model == "DSSM_CNN"):
    model = DSSM_biLSTM(conf)
elif (FLAGS.model == "DSSM_CNN"):
    model = DSSM_CNN(conf)
elif (FLAGS.model == "DSSM_CNN_ATTENTION"):
    model = DSSM_CNN_ATTENTION(conf)
elif (FLAGS.model == "Match_pyramid"):
    model = Match_pyramid(conf)
elif (FLAGS.model == "DSSM_biLSTM_attention"):
    model = DSSM_biLSTM_attention(conf)

model.set_embedding(loader.get_weights())
model.build()
model.train(train_data, train_label, test_data, test_label)
