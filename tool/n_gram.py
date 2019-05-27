#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import re
import sys
import numpy as np

def get_unigram(self, s):
    return list(s)

def get_bigram(self, s):
    bigram, index = [], 0
    while index < len(s) - 1:
        bigram.append(s[index: index + 2])
        index += 1
    return bigram

def get_trigram(self, s):
    trigram, index = [], 0
    while index < len(s) - 2:
        trigram.append(s[index: index + 3])
        index += 1
    return trigram

if __name__ == '__main__':
    file_name = sys.argv[1]

