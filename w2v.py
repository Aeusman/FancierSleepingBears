import collections
import os
import pandas
from six.moves import urllib
import zipfile
# import tensorflow as tf
# import keras
import re
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F
import torch.optim as optim


news_data = pandas.read_csv('uci-news-aggregator.csv')
print("Load news data")

# skip_list=['plosser','noyer','bunds','cooperman','urgest','4b',"didn't",'chipotle','djia','5th','direxion']
skip_list={}
contractions_dict = { 
"ain't":'is not',
"it'll":"it will",
"there'll":"there will",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd've": "he would have",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"i'd've": "I would have",
"i'd":'i would',
"i'm": "I am",
"i'll": "I will",
"she'll": "she will",
"he'll": "he will",
"i've": "I have",
"isn't": "is not",
"he'd":"he would",
"it'd've": "it would have",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd've": "she would have",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"that'd've": "that would have",
"there'd've": "there would have",
"they'd've": "they would have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what're": "what are",
"what've": "what have",
"when've": "when have",
"where'd": "where did",
"they'd": "they did",
"where've": "where have",
"it'd":"it would",
"may've":"may have",
"who've": "who have",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd've": "you would have",
"who'd":"who would",
"you'd":"you would",
"why'd":"why would",
"you'll": "you will",
"who'll": "who will",
"what'll": "what will",
"you're": "you are",
"you've": "you have"
}
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

# used_vocab = {}
# sent_len_count = {}
processed_data = []
vocabulary = []

for i, title in enumerate(news_data.TITLE):

    # Log progress
    if i % 10000 == 0:
        print(i, len(news_data.TITLE))

    # Replace contractions and split
    title = expand_contractions(title.lower())
    title = re.findall(r"[\w']+", title.lower())
#     processed_data.append(title)
#     vocabulary += title
#     processed=True
    
    # Too long of a sentence
#     if len(title) >= 20:
#         continue
        
#     if len(title) not in sent_len_count:
#         sent_len_count[len(title)] = 0

#     sent_len_count[len(title)] += 1

#     embedded_sent = torch.Tensor()
#     # Find embeddings
    sent = []
    for w in title:
        w = w.strip(', .:%\'')
        if w[-2:] == "'s":
            w = w[-2:]

        try:
            val = float(w)
            sent.append("<NUM>")
            continue
        except ValueError:
            pass
        sent.append(w)
    processed_data.append(sent)
    vocabulary += sent
    
#         if dictionary.get(w) is None:
#             if w not in skip_list:
#                 skip_list[w] = 0
#             skip_list[w] += 1
#         else:
        
#             if w not in used_vocab:
#                 used_vocab[w] = 0
#             used_vocab[w] += 1



def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

data, count, dictionary, reversed_dictionary = build_dataset(vocabulary, 1000000)


word2idx = dictionary
idx2word = reversed_dictionary
vocabulary = reversed_dictionary.values()
vocabulary_size = len(vocabulary)
tokenized_corpus = processed_data


window_size = 2
idx_pairs = []
# for each sentence
i = 0
for sentence in tokenized_corpus:
    i += 1
    if i % 10000 == 0:
        print(i, len(tokenized_corpus))
    indices = [word2idx[word] for word in sentence]
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array
print("Created idx pairs")


embedding_dims = 300
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dims)
        self.w = nn.Linear(embedding_dims, vocabulary_size, bias=False)
    def forward(self,x):
        return F.log_softmax(self.w(self.embedding(x)), dim=0)





embedding_dims = 300
# W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
model = Net()
# W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
num_epochs = 10

learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

for epo in range(num_epochs):
    loss_val = 0
    for i, (data, target) in enumerate(dataloader, 0):
        optimizer.zero_grad()
        x = Variable(data)
        y_true = Variable(target)
        
#         z1 = torch.matmul(W1, x)
#         z2 = torch.matmul(W2, z1)
    
#         log_softmax = F.log_softmax(z2, dim=0)
        log_softmax = model(x)
#         print(log_softmax.shape, y_true.shape)
        loss = F.nll_loss(log_softmax, y_true.view(-1))
        loss_val += loss.data[0]
        loss.backward()
#         W1.data -= learning_rate * W1.grad.data
#         W2.data -= learning_rate * W2.grad.data
        optimizer.step()
#         W1.grad.data.zero_()
#         W2.grad.data.zero_()
#         if i % 100 == 0:
        print(i)
#     if epo % 10 == 0:    
#         print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')

