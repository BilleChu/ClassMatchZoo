#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import sys
import numpy as np
sys.path.append("..")
from .basic_loader import *
from sklearn.model_selection import train_test_split
from scripts.lm.bert.tokenization import FullTokenizer, convert_to_unicode 
import tensorflow as tf

class BertDataLoader(BasicLoader):
    def __init__(self):
        super(DataLoader, self).__init__()
        self.classes = 0
        self.max_seq_length = 250

    def set_params(self, params):
        super(DataLoader, self).set_params(params)
        if ("max_seq_length" in params):
            self.max_seq_length = params["max_seq_length"]
        else:
            print ("max_length default: ", self.max_len_a)

    def get_categories(self):
        return self.categories

    def build(self, conf_file):
        with open(conf_file, "r") as fi:
            files = fi.readlines()
            self.all_input_ids = []
            self.all_input_mask = []
            self.all_segment_ids = []
            self.all_label_ids = []
            self.classes = len(files)
            self.categories = []
            classid = 0
            for line in files:
                vs = line.strip().split()
                classname = vs[0]
                self.categories.append(classname)
                filename = vs[1]
                print(classname, filename)
                with open(filename, 'r') as sample_file:
                    print(filename, "open success!!!")
                    for line in sample_file:
                        convert_single_example(line.strip())
                        label = [0.0] * self.classes # 2 classes
                        label[int(classid)] = 1.0
                        self.all_label_ids.append(label)
                classid += 1

    def input_fn_builder(self, features, is_training, drop_remainder):
      """Creates an `input_fn` closure to be passed to TPUEstimator."""
      def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features[3])

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    features[0], shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    features[1],
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    features[2],
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(features[3], shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
          d = d.repeat()
          d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

      return input_fn

    def load_dict(self, path):
        self.tokenizer = tokenization.FullTokenizer(vocab_file=path, do_lower_case=True)

    def convert_single_example(self, line):

        lines = lines.split("\t")
        tokens_a = convert_to_unicode(lines[0])
        tokens_a = self.tokenizer.tokenize(tokens_a)
        tokens_b = None
        if 2 == len(lines):
            tokens_b = convert_to_unicode(lines[1])
            tokens_b = self.tokenizer.tokenize(tokens_b)
        if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
        # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
        input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        self.all_input_ids.append(input_ids)
        self.all_input_mask.append(input_mask)
        self.all_segment_ids.append(segment_ids)

    def get_train_num(self):
        return self.train_size
        
    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
      """Truncates a sequence pair in place to the maximum length."""
      while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
          break
        if len(tokens_a) > len(tokens_b):
          tokens_a.pop()
        else:
          tokens_b.pop()

    def get_train_test(self):
        print ("Class Num : ", self.classes)
                    self.all_input_ids = []
            self.all_input_mask = []
            self.all_segment_ids = []
            self.all_label_ids = 
        train_inputs, test_inputs,\
        train_mask, test_mask,\
        train_segment, test_segment,\
        train_label, test_label = train_test_split (self.all_input_ids, \
                                                    self.all_input_mask, \
                                                    self.all_segment_ids, \
                                                    self.all_label_ids,\
                                                    test_size=self.ratio)
        self.train_size = len(train_label)
        self.test_size = len(test_label)
        return self.input_fn_builder([train_inputs, train_mask, train_segment, train_label], True, True), \
               self.input_fn_builder([test_inputs, test_mask, test_segment, test_label], False, False)

if __name__ == '__main__':
    conf_file = sys.argv[1]
    m = BertDataLoader()
    m.load_dict("./a")
    m.build(conf_file)