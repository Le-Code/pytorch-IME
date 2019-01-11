# -*- coding: utf-8 -*-

import torch
import random
from train import indexesFromSentence
from load import SOS_token, EOS_token
from load import MAX_LENGTH, loadPrepareData, Voc
from model import *
import os
from config import save_dir
import matplotlib
from matplotlib.font_manager import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 定义自定义字体，文件名从1.b查看系统中文字体中来
myfont = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
# 解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class Sentence:
    def __init__(self, decoder_hidden, last_idx=SOS_token, sentence_idxes=[], sentence_scores=[]):
        if(len(sentence_idxes) != len(sentence_scores)):
            raise ValueError("length of indexes and scores should be the same")
        self.decoder_hidden = decoder_hidden
        self.last_idx = last_idx
        self.sentence_idxes =  sentence_idxes
        self.sentence_scores = sentence_scores

    def avgScore(self):
        if len(self.sentence_scores) == 0:
            raise ValueError("Calculate average score of sentence, but got no word")
        # return mean of sentence_score
        return sum(self.sentence_scores) / len(self.sentence_scores)

    def addTopk(self, topi, topv, decoder_hidden, beam_size, voc):
        topv = torch.log(topv)
        terminates, sentences = [], []
        for i in range(beam_size):
            if topi[0][i] == EOS_token:
                terminates.append(([voc.index2word[idx.item()] for idx in self.sentence_idxes] + ['<EOS>'],
                                   self.avgScore())) # tuple(word_list, score_float
                continue
            idxes = self.sentence_idxes[:] # pass by value
            scores = self.sentence_scores[:] # pass by value
            idxes.append(topi[0][i])
            scores.append(topv[0][i])
            sentences.append(Sentence(decoder_hidden, topi[0][i], idxes, scores))
        return terminates, sentences

    def toWordScore(self, voc):
        words = []
        for i in range(len(self.sentence_idxes)):
            if self.sentence_idxes[i] == EOS_token:
                words.append('<EOS>')
            else:
                words.append(voc.index2word[self.sentence_idxes[i].item()])
        if self.sentence_idxes[-1] != EOS_token:
            words.append('<EOS>')
        return (words, self.avgScore())

def beam_decode(decoder, decoder_hidden, encoder_outputs, voc, beam_size, max_length=MAX_LENGTH):
    terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
    prev_top_sentences.append(Sentence(decoder_hidden))
    for i in range(max_length):
        for sentence in prev_top_sentences:
            decoder_input = torch.LongTensor([[sentence.last_idx]])
            decoder_input = decoder_input.to(device)

            decoder_hidden = sentence.decoder_hidden
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.topk(beam_size)
            term, top = sentence.addTopk(topi, topv, decoder_hidden, beam_size, voc)
            terminal_sentences.extend(term)
            next_top_sentences.extend(top)

        next_top_sentences.sort(key=lambda s: s.avgScore(), reverse=True)
        prev_top_sentences = next_top_sentences[:beam_size]
        next_top_sentences = []

    terminal_sentences += [sentence.toWordScore(voc) for sentence in prev_top_sentences]
    terminal_sentences.sort(key=lambda x: x[1], reverse=True)

    n = min(len(terminal_sentences), 15)
    return terminal_sentences[:n]

def decode(decoder, decoder_hidden, encoder_outputs, voc,input_len, max_length=MAX_LENGTH):

    decoder_input = torch.LongTensor([[SOS_token]])
    decoder_input = decoder_input.to(device)

    decoded_words = []
    decoder_attentions = torch.zeros(MAX_LENGTH, input_len) #TODO: or (MAX_LEN+1, MAX_LEN+1)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attn = torch.squeeze(decoder_attn,0)
        print(decoder_attn.data)
        decoder_attentions[di] = decoder_attn.data
        _, topi = decoder_output.topk(3)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(voc.index2word[ni.item()])

        decoder_input = torch.LongTensor([[ni]])
        decoder_input = decoder_input.to(device)

    return decoded_words, decoder_attentions[:di+1]


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with color bar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90,fontproperties=myfont)
    ax.set_yticklabels([''] + output_words,fontproperties=myfont)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluate(encoder, decoder,pinyin_voc,word_voc, sentence, beam_size, max_length=MAX_LENGTH):
    indexes_batch = [indexesFromSentence(pinyin_voc, sentence)] #[1, seq_len]
    lengths = [len(indexes) for indexes in indexes_batch]
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)

    encoder_outputs, encoder_hidden = encoder(input_batch, lengths, None)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    if beam_size == 1:
        return decode(decoder, decoder_hidden, encoder_outputs, word_voc,len(indexes_batch[0]))
    else:
        return beam_decode(decoder, decoder_hidden, encoder_outputs, word_voc, beam_size)


def evaluateRandomly(encoder, decoder, voc, pairs, reverse, beam_size, n=10):
    for _ in range(n):
        pair = random.choice(pairs)
        print("=============================================================")
        if reverse:
            print('>', " ".join(reversed(pair[0].split())))
        else:
            print('>', pair[0])
        if beam_size == 1:
            output_words, _ = evaluate(encoder, decoder, voc, pair[0], beam_size)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
        else:
            output_words_list = evaluate(encoder, decoder, voc, pair[0], beam_size)
            for output_words, score in output_words_list:
                output_sentence = ' '.join(output_words)
                print("{:.3f} < {}".format(score, output_sentence))

def evaluateInput(encoder, decoder, pinyin_voc,word_voc, beam_size):
    pair = ''
    while(1):
        try:
            pair = input('> ')
            if pair == 'q': break
            if beam_size == 1:
                output_words, attentions = evaluate(encoder, decoder,pinyin_voc,word_voc, pair, beam_size)
                showAttention(pair,output_words,attentions)
                output_sentence = ' '.join(output_words)
                print('<', output_sentence)
            else:
                output_words_list = evaluate(encoder, decoder, voc, pair, beam_size)
                for output_words, score in output_words_list:
                    output_sentence = ' '.join(output_words)
                    print("{:.3f} < {}".format(score, output_sentence))
        except KeyError:
            print("Incorrect spelling.")



def runTest(n_layers, hidden_size, reverse, modelFile, beam_size, inp, corpus):
    torch.set_grad_enabled(False)

    voc, pairs = loadPrepareData(corpus)
    embedding = nn.Embedding(voc.n_words, hidden_size)
    encoder = EncoderRNN(voc.n_words, hidden_size, embedding, n_layers)
    attn_model = 'dot'
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers)

    checkpoint = torch.load(modelFile)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    # train mode set to false, effect only on dropout, batchNorm
    encoder.train(False);
    decoder.train(False);

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    if inp:
        evaluateInput(encoder, decoder, voc, beam_size)
    else:
        evaluateRandomly(encoder, decoder, voc, pairs, reverse, beam_size, 20)

if __name__=='__main__':
    torch.set_grad_enabled(False)

    hidden_size = 512
    n_layers = 1
    attn_mode = 'dot'

    pinyin_voc,word_voc,pairs = loadPrepareData('data/touchpal_done.txt')
    pinyin_embedding = nn.Embedding(pinyin_voc.n_words,hidden_size)
    word_embedding = nn.Embedding(word_voc.n_words,hidden_size)
    encoder = EncoderRNN(pinyin_voc.n_words,hidden_size,pinyin_embedding,n_layers)
    decoder = LuongAttnDecoderRNN(attn_mode,word_embedding,hidden_size,word_voc.n_words,n_layers)

    checkpoint = torch.load('save/model/touchpal_done/1-1_512/6000_backup_bidir_model.tar')
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    encoder.train(False)
    decoder.train(False)

    evaluateInput(encoder,decoder,pinyin_voc,word_voc,1)
