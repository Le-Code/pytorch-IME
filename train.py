import torch
import torch.nn as nn
from torch import optim
import torch.backends.cudnn as cudnn

import itertools
import random
import math
import os
# tqdm是一个进度条工具
from tqdm import tqdm
from load import loadPrepareData
from load import SOS_token, EOS_token, PAD_token
from model import EncoderRNN, LuongAttnDecoderRNN,DecoderWithoutAttn
from config import MAX_LENGTH, teacher_forcing_ratio, save_dir

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

cudnn.benchmark = True
#############################################
# generate file name for saving parameters
#############################################
def filename(reverse, obj):
	filename = ''
	if reverse:
		filename += 'reverse_'
	filename += obj
	return filename


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

# 构建矩阵，l_pinyin表示候选拼音的集合,l_word 表示候选汉字的集合
# 构建one-hot矩阵，包含候选拼音和候选候选汉字，计算损失函数的时候会用到
def binaryMatrix(l_pinyin,l_word, value=PAD_token):
    m = []

    for i, seq in enumerate(l_pinyin):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    for i,seq in enumerate(l_word):
        for token in seq:
            if token==PAD_token:
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
def outputVar(l_pinyin,l_word,pinyin_voc,word_voc):
    indexes_batch_pinyin = [indexesFromSentence(pinyin_voc, sentence) for sentence in l_pinyin]
    indexes_batch_word = [indexesFromSentence(word_voc,sentence) for sentence in l_word]
    # 由于 word 和 拼音 的长度一致，所以两者随便取一个便可
    max_target_len = max([len(indexes) for indexes in indexes_batch_pinyin])
    padList_pinyin = zeroPadding(indexes_batch_pinyin)
    padList_word = zeroPadding(indexes_batch_word)
    mask = binaryMatrix(padList_pinyin,padList_word)
    mask = torch.ByteTensor(mask)
    padVar_pinyin = torch.LongTensor(padList_pinyin)
    padVar_word = torch.LongTensor(padList_word)
    return padVar_pinyin,padVar_word, mask, max_target_len

# pair_batch is a list of (input, output) with length batch_size
# sort list of (input, output) pairs by input length, reverse input
# return input, lengths for pack_padded_sequence, output_variable, mask
def batch2TrainData(pinyin_voc,word_voc, tuple_batch, reverse):
    if reverse:
        tuple_batch = [pair[::-1] for pair in tuple_batch]
    tuple_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch_pinyin,output_batch_word = [], [], []
    for tup in tuple_batch:
        input_batch.append(tup[0])
        output_batch_pinyin.append(tup[1])
        output_batch_word.append(tup[2])
    inp, lengths = inputVar(input_batch, word_voc)
    output_pinyin,output_word, mask, max_target_len = outputVar(output_batch_pinyin,output_batch_word,pinyin_voc,word_voc)
    return inp, lengths, output_pinyin,output_word, mask, max_target_len

#############################################
# Training
#############################################

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

def train(input_variable, lengths, target_variable_pinyin,target_variable_word, mask, max_target_len, encoder,encoder_sec, decoder,decoder_sec, pinyin_embedding,word_embedding,
          encoder_optimizer,encoder_sec_optimizer, decoder_optimizer,decoder_sec_optimizer, batch_size, max_length=MAX_LENGTH):

    encoder_optimizer.zero_grad()
    encoder_sec_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    decoder_sec_optimizer.zero_grad()


    # input_variable = input_variable.to(device)
    # target_variable = target_variable.to(device)
    # mask = mask.to(device)

    loss = 0
    print_losses = []
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(input_variable, lengths, None)

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
            # decoder_input = decoder_input.to(device)
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

    pinyin_voc,word_voc, tuples = loadPrepareData(corpus)

    # training data
    corpus_name = os.path.split(corpus)[-1].split('.')[0]
    training_batches = None
    try:
        training_batches = torch.load(os.path.join(save_dir, 'training_data', corpus_name,
                                                   '{}_{}_{}.tar'.format(n_iteration, \
                                                                         filename(reverse, 'training_batches'), \
                                                                         batch_size)))
    except FileNotFoundError:
        print('Training pairs not found, generating ...')
        training_batches = [batch2TrainData(pinyin_voc,word_voc, [random.choice(tuples) for _ in range(batch_size)], reverse)
                          for _ in range(n_iteration)]
        torch.save(training_batches, os.path.join(save_dir, 'training_data', corpus_name,
                                                  '{}_{}_{}.tar'.format(n_iteration, \
                                                                        filename(reverse, 'training_batches'), \
                                                                        batch_size)))
    # model
    checkpoint = None
    print('Building encoder and decoder ...')
    pinyin_embedding = nn.Embedding(pinyin_voc.n_words, hidden_size)
    word_embedding = nn.Embedding(word_voc.n_words,hidden_size)
    # 第一层Encoder,解码汉字
    encoder = EncoderRNN(word_voc.n_words, hidden_size, word_embedding, n_layers, dropout)
    # 构建第二层Encoder，解码拼音
    encoder_second = EncoderRNN(pinyin_voc.n_words,hidden_size,pinyin_embedding,n_layers,dropout)
    attn_model = 'dot'
    # 第一层decoder，解析拼音，基于注意力
    decoder = LuongAttnDecoderRNN(attn_model, pinyin_embedding, hidden_size, pinyin_voc.n_words, n_layers, dropout)
    # 构建第二层Decoder，解析汉字 ，先暂时不用注意力模型
    decoder_second = DecoderWithoutAttn(word_embedding,hidden_size,word_voc.n_words,n_layers,dropout)
    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder.load_state_dict(checkpoint['en'])
        encoder_second.load_state_dict(checkpoint['en_sec'])
        decoder.load_state_dict(checkpoint['de'])
        decoder_second.load_state_dict(checkpoint['de_sec'])
    # use cuda
    # encoder = encoder.to(device)
    # decoder = decoder.to(device)

    # optimizer
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    encoder_second_optimizer = optim.Adam(encoder_second.parameters(),lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    decoder_second_optimizer = optim.Adam(decoder_second.parameters(),lr=learning_rate*decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        encoder_second_optimizer.load_state_dict(checkpoint['en_sec_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])
        decoder_second_optimizer.load_state_dict(checkpoint['de_sec_opt'])

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
        input_variable, lengths, target_variable_pinyin,target_variable_word, mask, max_target_len = training_batch

        loss = train(input_variable, lengths, target_variable_pinyin,target_variable_word, mask, max_target_len, encoder,
                     decoder, pinyin_embedding,word_embedding, encoder_optimizer, decoder_optimizer, batch_size)
        print_loss += loss
        perplexity.append(loss)

        if iteration % print_every == 0:
            print_loss_avg = math.exp(print_loss / print_every)
            print('%d %d%% %.4f' % (iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, 'model', corpus_name, '{}-{}_{}'.format(n_layers, n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'en_sec':encoder_second.state_dict(),
                'de': decoder.state_dict(),
                'de_sec':decoder_second.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'en_sec_opt':encoder_second_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'de_sec_opt':decoder_second_optimizer.state_dict(),
                'loss': loss,
                'plt': perplexity
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, filename(reverse, 'backup_bidir_model'))))
