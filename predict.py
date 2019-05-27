#! /usr/bin python
import sys
import os
import numpy as np
sys.path.append("./scripts/")
from scripts.classification.textcnn import TextCnn
from data.data_loader import DataLoader

conf_file = sys.argv[1]
loader = DataLoader()
loader.load_dict("data/title_dict.json")
loader.build(conf_file)
data = loader.get_predict()
conf2 = {"embedding_size": loader.word_vec_len,
         "vocab_size": len(loader.weights),
         "sequence_len": loader.max_len,
         "epochs": 10}
model = TextCnn(conf2)
model.load_model("model/weights.010-0.9952.hdf5")
result = model.predict(data)
f = open("/mnt/hgfs/share/pornCensor/query.sug.title/query.random.5w", "r")
fw = open("neg", "w")
lines = f.readlines()
for i in range(result.shape[0]):
    if result[i][1] > 0.5:
        print (lines[i])
    else:
        fw.write(lines[i])
fw.close()
f.close()
