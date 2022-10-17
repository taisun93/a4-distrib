# models.py

import string
from tokenize import String
import numpy as np
import collections
import torch.nn as nn
import torch
from torch import optim
import torch.nn.functional as F

#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1

class LSTM(nn.Module):
    def __init__(self, vocab_index):
        hidden = 5
        input = 8
        super(LSTM, self).__init__()
        self.vocab_index = vocab_index
        self.embedLayer = nn.Embedding(27, input)

        self.LSTM = nn.LSTM(input, hidden)

        self.W = nn.Linear(hidden, hidden)
        self.W2 = nn.Linear(hidden, 2)
        # self.W2 = nn.Linear(2, 1)

        for layer in self.LSTM._all_weights:
            for att in layer:
                if 'weight' in att:
                    nn.init.xavier_uniform_(self.LSTM.__getattr__(att))
        

    def run(self, input):
        ex_converted = []
        for x in input: ex_converted.append(self.vocab_index.index_of(x))
        
        ts = torch.LongTensor(ex_converted)
        embedded_input = self.embedLayer(ts)
        

        out, _ = self.LSTM(embedded_input)
        # print(type(self.log_softmax))
        
        

        # final = self.W2(final)
        final1 = self.W(out[-1])
        return self.W2(final1)
        # return F.log_softmax(final.view(2,-1), dim=1)

        

class RNNClassifier(ConsonantVowelClassifier):
    
    def __init__(self, vocab_index):
        # self.embed = embed
        self.NN = LSTM(vocab_index)
    
    def getprob(self, context):
        return self.NN.run(context)

    def predict(self, context):
        answer = self.NN.run(context)
             
        if answer[0]>answer[1]:
            return 0
        return 1


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """

    train_cons = np.transpose((np.array(train_cons_exs), np.zeros((len(train_cons_exs)))))
    train_vows = np.transpose((np.array(train_vowel_exs), np.ones((len(train_vowel_exs)))))
    train_exs = np.vstack((train_cons, train_vows))
    

    classifier = RNNClassifier(vocab_index)
    # train_exs = train_exs[:40]

    # fst = train_exs[0][0]
    # prob = classifier.getprob(fst)
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.NN.parameters(), lr=.002)
    # print(prob)
    for epoch in range(6):
        np.random.shuffle(train_exs)
        totalLoss = 0
        # random.shuffle(train_exs)
        for ex in train_exs:
            words, target = ex[0], float(ex[1])
            # print(words)
            # print(target)
            # print(type(target))
            prob = classifier.getprob(words)
            
            if target == 0.0:
                correct = torch.tensor([1.0, 0.0])
            elif target == 1.0:
                correct = torch.tensor([0.0, 1.0])


            loss = lossFunction(prob,correct)
            # print(loss)
            totalLoss+=loss
            classifier.NN.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"loss in {epoch} is {totalLoss}")

    return classifier
    # ex_converted = []

    
    # for x in fst: ex_converted.append(vocab_index.index_of(x))
        
    # print(len(ex_converted))
    # fst_exp = np.transpose((np.array(range(0,20)), np.array(ex_converted)))
    # # print(type(train_exs[0][0]))
    
    # print(convert_tensor(fst))
    # print(ex_converted)

    # ts = torch.LongTensor(ex_converted)
    
    # print(embedLayer(ts))

    # print(train_exs[:5])

    # dev_cons = np.transpose((np.array(dev_cons_exs), np.zeros((len(dev_cons_exs)))))
    # dev_vows = np.transpose((np.array(dev_vowel_exs), np.ones((len(dev_vowel_exs)))))


#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self):
        raise Exception("Implement me")

    def get_next_char_log_probs(self, context):
        raise Exception("Implement me")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Implement me")


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    print(train_text[0:10])

