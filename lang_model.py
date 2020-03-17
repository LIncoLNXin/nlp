# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as T
import config


class LanguageModel(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, dropout = 0.2):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional = bidirectional, dropout = dropout, batch_first = True)

		self.num_directions = 2 if bidirectional else 1 
		self.fc = nn.Linear(hidden_dim * self.num_directions, vocab_size)
		self.dropout = nn.Dropout(dropout)
		
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.vocab_size = vocab_size

	def init_hidden(self, batch_size):
		return (torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_dim), 
				torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_dim))

	def forward(self, text, mask):
		embedded = self.embedding(text)
		embedded = self.dropout(embedded)   # batch_size, max_seq_len, embedding_dim
		
		seq_lengths = (1 - mask).sum(1)
		embedded = T.rnn.pack_padded_sequence(embedded, lengths = seq_lengths, batch_first = True, enforce_sorted = False)
		
		hidden = self.init_hidden(text.size(0))
		output, hidden = self.rnn(embedded, hidden)

		output, _ = T.rnn.pad_packed_sequence(output, batch_first = True, 
				padding_value = config.padding_value, total_length = mask.shape[1])

		output = self.dropout(output)
		
		# output  batch_size, seq_len, hidden_size * num_directions
		# rets 	  batch_size, seq_len, vocab_size
		rets = self.fc(output)
		# return rets		这里 是为了 让 外部 不关心 vocab_size
		return rets.view(-1, self.vocab_size)
		
		
				
				
				



				
