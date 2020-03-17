# -*- coding: utf-8 -*-

import random
random.seed(1234)

import config

def generate_word_dict(filename):
	word_to_index = {}
	index_to_word = []

	index_to_word.append(config.PAD)
	word_to_index[config.PAD] = config.PAD_INDEX			# pad 0

	with open(filename, "r", encoding='utf-8') as f:
		index = 1
		for line in f:
			line = line.strip()
			if not line:
				continue
			line = line.lower()
			if line in word_to_index:
				continue
			word_to_index[line] = index
			index_to_word.append(line)
			index += 1
	return word_to_index, index_to_word
	
def process_data(filename, word_to_index):
	word_vecs = []
	with open(filename, "r", encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			line = line.split(' ')
			word_vec = [word_to_index.get(word.lower(), 0) for word in line if word.strip()]
			word_vecs.append(word_vec)
	return word_vecs

def generate_batch(word_vecs, batch_size):
	n_chunk = len(word_vecs) // batch_size
	left_size = len(word_vecs) - n_chunk * batch_size
	if left_size > 0:
		n_chunk += 1
	
	x_batches = []
	y_batches = []
	
	random.shuffle(word_vecs)
	
	for i in range(n_chunk):
		start = i * batch_size
		end = start + batch_size
		
		if end > len(word_vecs):
			batches = word_vecs[start:]
		else:
			batches = word_vecs[start:end]
			
		max_len = max(map(len, batches))
		x_data = []
		y_data = []
		for row in range(len(batches)):
			row_len = len(batches[row])
			cur_x = [config.PAD] * (max_len - 1)
			cur_y = [config.PAD] * (max_len - 1)
			cur_x[:row_len - 1] = batches[row][:row_len - 1]
			cur_y[:row_len - 1] = batches[row][1:]
			x_data.append(cur_x)
			y_data.append(cur_y)
			
		x_batches.append(x_data)
		y_batches.append(y_data)
		
	return x_batches, y_batches
	





