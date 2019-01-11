import torch
from load import loadPrepareData
from load import EOS_token,SOS_token,PAD_token
import itertools
import os
import random
from tqdm import tqdm
import math
import torch.nn as nn
from model import EncoderRNN
from model import LuongAttnDecoderRNN
from torch import optim

from config import MAX_LENGTH
from config import teacher_forcing_ratio

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

#############################################
# Prepare Training Data
#############################################
#讲一个句子用index来表示
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

# batch_first: true -> false, i.e. shape: seq_len * batch
def zeroPadding(l, fillvalue=PAD_token):
    # 和zip函数类似，形成一一对应的集合，差别在于zip以最少为基准，这个函数以最多为基准，对于缺少的用fillvalue来代替
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

# 构建矩阵，l表示seq的集合
def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# convert to index, add EOS
# return input pack_padded_sequence
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = [len(indexes) for indexes in indexes_batch]
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# convert to index, add EOS, zero padding
# return output variable, mask, max length of the sentences in batch
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# pair_batch is a list of (input, output) with length batch_size
# sort list of (input, output) pairs by input length, reverse input
# return input, lengths for pack_padded_sequence, output_variable, mask
def batch2TrainData(voc, pair_batch, reverse):
    if reverse:
        pair_batch = [pair[::-1] for pair in pair_batch]
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


def load(filename,reverse=False,n_iteration=1,batch_size=64):
    voc, pairs = loadPrepareData(filename)

    # training data
    corpus_name = os.path.split(filename)[-1].split('.')[0]
    training_batches = None
    print('Training pairs not found, generating ...')
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)], reverse)
                            for _ in range(n_iteration)]
    print(len(training_batches))
    print(training_batches)


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, max_length=MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    loss = 0
    print_losses = []
    n_totals = 0


    encoder_outputs, encoder_hidden = encoder(input_variable, lengths, None)

    #
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Run through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_input = target_variable[t].view(1, -1) # Next input is current target
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal) #v0.4
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            _, topi = decoder_output.topk(1) # [64, 1]

            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    loss.backward()

    clip = 50.0
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(corpus, reverse, n_iteration, learning_rate, batch_size, n_layers, hidden_size,
                print_every, save_every, dropout, loadFilename=None, attn_model='dot', decoder_learning_ratio=5.0):

    voc, pairs = loadPrepareData(corpus)

    # training data
    corpus_name = os.path.split(corpus)[-1].split('.')[0]
    training_batches = None
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)], reverse)
                          for _ in range(n_iteration)]
    # model
    checkpoint = None
    print('Building encoder and decoder ...')
    embedding = nn.Embedding(voc.n_words, hidden_size)
    encoder = EncoderRNN(voc.n_words, hidden_size, embedding, n_layers, dropout)
    attn_model = 'dot'
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers, dropout)
    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])
    # use cuda
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # optimizer
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])

    # initialize
    print('Initializing ...')
    start_iteration = 1
    perplexity = []
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1
        perplexity = checkpoint['plt']
    # 进度条显示
    for iteration in tqdm(range(start_iteration, n_iteration + 1)):
        # 得到当前iteration的数据
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size)
        print_loss += loss
        perplexity.append(loss)


if __name__=='__main__':
    trainIters('data/movie_subtitles.txt',False,1,0.01,64,1,256,1,1,0.1)
