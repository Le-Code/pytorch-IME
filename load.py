import torch
import re
import os
import unicodedata

from config import MAX_LENGTH, save_dir

SOS_token = 0
EOS_token = 1
PAD_token = 2

class Voc:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"PAD"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readVocs(corpus, corpus_name):
    print("Reading lines...")

    # combine every two lines into pairs and normalize
    with open(corpus) as f:
        content = f.readlines()
    # import gzip
    # content = gzip.open(corpus, 'rt')
    lines = [x.strip() for x in content]
    it = iter(lines)
    # pairs = [[normalizeString(x), normalizeString(next(it))] for x in it]
    #[question,reply_pinyin,reply_word]
    tuples = [[x.split(':')[0]]+[s for s in next(it).split(':')[::-1]] for x in it]

    pinyin_voc,word_voc = Voc("pinyin"),Voc('word')
    return pinyin_voc,word_voc, tuples

def filterTup(p):
    # input sequences need to preserve the last word for EOS_token
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and len(p[1].split(' '))==len(p[2].split(' '))

def filterTuples(tuples):
    return [tup for tup in tuples if filterTup(tup)]

def prepareData(corpus, corpus_name):
    pinyin_voc,word_voc, tuples = readVocs(corpus, corpus_name)
    print("Read {!s} sentence tuples".format(len(tuples)))
    tuples = filterTuples(tuples)
    print("Trimmed to {!s} sentence tuples".format(len(tuples)))
    print("Counting words...")
    for tup in tuples:
        word_voc.addSentence(tup[0])
        pinyin_voc.addSentence(tup[1])
        word_voc.addSentence(tup[2])
    print("Counted words:", word_voc.n_words)
    print('counted pinyin:',pinyin_voc.n_words)
    directory = os.path.join(save_dir, 'training_data', corpus_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(word_voc, os.path.join(directory, '{!s}.tar'.format('voc_word')))
    torch.save(pinyin_voc, os.path.join(directory, '{!s}.tar'.format('voc_pinyin')))
    torch.save(tuples, os.path.join(directory, '{!s}.tar'.format('tuples')))
    return pinyin_voc,word_voc, tuples

def loadPrepareData(corpus):
    corpus_name = corpus.split('/')[-1].split('.')[0]
    try:
        print("Start loading training data ...")
        pinyin_voc = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'voc_pinyin.tar'))
        word_voc = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'voc_word.tar'))
        tuples = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'tuples.tar'))
    except FileNotFoundError:
        print("Saved data not found, start preparing trianing data ...")
        pinyin_voc,word_voc, tuples = prepareData(corpus, corpus_name)
    return pinyin_voc,word_voc, tuples
