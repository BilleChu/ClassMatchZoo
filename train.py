#! /usr/bin python
import os
import sys
import argparse
import numpy as np

sys.path.append("./scripts/")
from scripts.classification.lr                  import Lr
from scripts.classification.textcnn             import TextCnn
from scripts.classification.fasttext            import Fasttext
from scripts.classification.textrnn             import TextRnn
from scripts.classification.textrnn_attention   import AttentiveTextRnn
from data.data_loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_file",
    type=str,
    default="data/query_conf",
    help="conf_file containes sample files and labels")

parser.add_argument(
    "--w2v_path",
    type=str,
    default="/mnt/hgfs/share/pornCensor/query.skip.vec.win3",
    help="w2v file which provide w2v")
FLAGS, unparsed = parser.parse_known_args()
print ("unparsed: ", unparsed)


params =  {"ratio": 0.2,
            "max_len": 15,
            "embedding_size": 100}
loader = DataLoader()
loader.set_params(params)
loader.set_w2v(FLAGS.w2v_path)
loader.build(FLAGS.conf_file)
#loader.save_dict("data/title_dict.json")
train_data, test_data, train_label, test_label = loader.get_train_test()


conf = {"embedding_size": loader.word_vec_len,
         "vocab_size": len(loader.weights),
         "sequence_len": loader.max_len,
         "epochs": 100,
         "classes": loader.classes}
#model  = Lr(conf)
model = TextCnn(conf)
#model = Fasttext(conf)
#model = TextRnn(conf)
#model = AttentiveTextRnn(conf)
model.set_embedding(loader.get_weights())
model.set_categories(loader.get_categories())
model.build()
model.train(train_data, train_label, test_data, test_label)
