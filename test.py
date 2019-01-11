# coding:utf-8
import matplotlib
import itertools
import os
import torch
from load import PAD_token
import matplotlib.pyplot as plt
from matplotlib.font_manager import *

def readVocs():
    with open('/home/jerry/Documents/jerry/develop/python/pytorch-chatbot/data/movie_subtitles.txt') as f:
        content = f.readlines()

    lines = [x.strip() for x in content]

    it = iter(lines)

    pair = [[x,next(it)] for x in it]

    return pair

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def splitCorpus(corpus):
    corpus_name = os.path.split(corpus)[-1].split(".")[0]
    return corpus_name

def testInputBatch():
    s = [['request1','answer1'],['request2','answer2']]
    input_batch = []
    for pair in s:
        input_batch.append(pair[0])
    return input_batch

def f():
    a = [1,2,3,4]
    return a,10

def pack_padded_sequence():
    a = torch.Tensor([[1,2,3,4,5],[6,7,8,-1,-1],[9,10,-1,-1,-1],[11,-1,-1,-1,-1]])
    length = [4,3,2,1,1]
    packed = torch.nn.utils.rnn.pack_padded_sequence(a,lengths=length)
    print(packed.data)
    print(packed.batch_sizes)

if __name__=="__main__":
    pack_padded_sequence()