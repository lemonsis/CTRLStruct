# from tkinter import _Padding
# from unittest.util import _MAX_LENGTH
import torch.nn as nn
from transformers import BartConfig, BartTokenizer
from transformers.models.bart.modeling_bart import BartEncoder, BartForSequenceClassification
import torch

def mean_pooling(input):
    sentence = torch.sum(input, dim = 1) / input.shape[1]
    return sentence

class Model(nn.Module):
    def __init__(self, pretrained_model='facebook/bart-large', dropout_augment=False, cls=False, dropout_prob=0.2):
        super().__init__()
        conf = BartConfig.from_pretrained(pretrained_model)
        if dropout_augment:
            conf.attention_dropout = dropout_prob
            conf.dropout = dropout_prob
        self.cls = cls
        # if self.cls:
        #     self.encoder = BartForSequenceClassification.from_pretrained(pretrained_model, config=conf)
        self.encoder = BartEncoder.from_pretrained(pretrained_model, config=conf)

    def forward(self, input_ids, attention_mask, output_attentions=None, output_hidden_states=None, return_dict=True):
        # if self.cls:
        #     ouput = self.encoder
        output = self.encoder(input_ids, attention_mask)
        output = output.last_hidden_state
        output = mean_pooling(output)
        return output

# model = BartEncoder.from_pretrained('facebook/bart-large')
# input = "We can do it ."
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
# encode_input = tokenizer(input, return_tensors='pt', max_length = 64, padding='max_length', truncation=True)
# print(model(**encode_input).last_hidden_state.shape)