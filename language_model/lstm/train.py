# -*- coding: utf-8 -*-
import torch
import data_loader
import lang_model 
import config
import time
import sys
import collections


# 获取 预测结果 对应的 word下标
def get_pred_indices(pred, batch_size):
	# pred 	[batch_size * max_seq_len, vocab_size]

	pred = pred.view(batch_size, -1, pred.size(1))
	softmax = torch.nn.Softmax(dim = 2)		
	pred = softmax(pred)
	values, indices = torch.max(pred, dim = 2)	# pred中pad的位置填充的是一个很小的值, 不影响softmax之后概率的大小关系

	# indices [batch_size, max_seq_len]  预测的结果index
	return indices


def binary_accuracy(pred, label, mask):
	# pred 	[batch_size * max_seq_len, vocab_size]
	# label [batch_size, max_seq_len]
	# mask [batch_size, max_seq_len]		pad的地方是1， 其他是0

	indices = get_pred_indices(pred, label.size(0))
	correct = (indices == label)

	correct = correct * (1 - mask)	# 不关心 pad位置的结果
	acc = correct.float().sum() / (1 - mask).sum()  # 不统计pad位置 
	return acc

# 输出 错误答案 和 对应标准答案
def get_error_words(pred, label, mask, index_to_word):
	indices = get_pred_indices(pred, label.size(0))
	errors = 1 - (indices == label)	
	
	errors = errors * (1 - mask) # 不关心 pad位置的结果
	
	err_pred = torch.masked_select(indices, errors)
	err_label = torch.masked_select(label, errors)

	rets = [] 
	for i in range(len(err_pred)):
		pred_word = index_to_word[err_pred[i].item()]
		label_word = index_to_word[err_label[i].item()]
		#print("label_word:", label_word, "pred_word:", pred_word)
		rets.append((label_word, pred_word))
	return rets


# 统计最常见的topK错误 及 次数
def get_top_error_words(err_words, topK):
	ws = dict(collections.Counter(err_words))
	ws = sorted(ws.items(), key = lambda x: -x[1])
	#ws, _ = zip(*ws)

	max_num = topK if topK < len(ws) else len(ws)
	return ws[:max_num]
	

def train(model, batches, optimizer, criterion):
	epoch_loss = 0
	epoch_acc = 0
	
	model.train()
	for batch in batches:
		data = torch.tensor(batches[0])		# batch_size, max_seq_len
		label = torch.tensor(batches[1])	# batch_size, max_seq_len
		
		mask = data == config.PAD_INDEX
	
		optimizer.zero_grad()
		
		model.zero_grad()
		pred = model(data, mask)	
		loss = criterion(pred, label.view(-1))
		acc = binary_accuracy(pred, label, mask)
		loss.backward()
			
		optimizer.step()
		
		epoch_loss += loss.item()
		epoch_acc += acc.item()
		
	return epoch_loss / len(batches), epoch_acc / len(batches)


def evaluate(model, batches, criterion, index_to_word = None):
	epoch_loss = 0
	epoch_acc = 0
	error_words = []
	
	model.eval()
	with torch.no_grad():
		for batch in batches:
			data = torch.tensor(batches[0])
			label = torch.tensor(batches[1])
			
			mask = data == config.PAD_INDEX
	
			pred = model(data, mask)
			loss = criterion(pred, label.view(-1))
			acc = binary_accuracy(pred, label, mask)
			
			epoch_loss += loss.item()
			epoch_acc += acc.item()

			if index_to_word is not None:
				error_words.extend( get_error_words(pred, label, mask, index_to_word))
			
	return epoch_loss / len(batches), epoch_acc / len(batches), error_words
	

def run():
	word_to_index, index_to_word = data_loader.generate_word_dict(config.word_dict_file)
	vocab_size = len(word_to_index)
	print("vocab_size: ", vocab_size)
	print("index_to_word size:", len(index_to_word))

	train_vecs = data_loader.process_data(config.train_file, word_to_index)
	dev_vecs = data_loader.process_data(config.dev_file, word_to_index)
	test_vecs = data_loader.process_data(config.test_file, word_to_index)

	print("train_vecs size:", len(train_vecs))
	print("dev_vecs size:", len(dev_vecs))
	print("test_vecs size:", len(test_vecs))

	train_x_batches, train_y_batches = data_loader.generate_batch(train_vecs, config.batch_size)
	dev_x_batches, dev_y_batches = data_loader.generate_batch(dev_vecs, config.batch_size)
	test_x_batches, test_y_batches = data_loader.generate_batch(test_vecs, config.batch_size)

	train_batches = list(zip(train_x_batches, train_y_batches))
	dev_batches = list(zip(dev_x_batches, dev_y_batches))
	test_batches = list(zip(test_x_batches, test_y_batches))

	print("train_batches size:", len(train_batches))
	print("dev_batches:", len(dev_batches))
	print("test_batches:", len(test_batches))
	
	model = lang_model.LanguageModel(vocab_size, config.embedding_dim, config.hidden_dim, config.num_layers, config.bidirectional, config.dropout)
	criterion = torch.nn.CrossEntropyLoss(ignore_index = config.padding_value) 	# loss 不考虑pad位置填充的值
	optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.learning_rate_gamma)

	best_valid_loss = float('inf')
	best_valid_acc = 0.0
	for epoch in range(config.epoch_num):
		print("start epoch ", epoch)
		start_time = time.time()
		
		train_loss = 0
		train_acc = 0
		for batch in train_batches:
			loss, acc = train(model, batch, optimizer, criterion)
			train_loss += loss
			train_acc += acc
		train_loss = train_loss / len(train_batches)
		train_acc = train_acc / len(train_batches)

		valid_loss = 0
		valid_acc = 0
		for batch in dev_batches:
			loss, acc, _ = evaluate(model, batch, criterion)
			valid_loss += loss
			valid_acc += acc
		valid_loss = valid_loss / len(dev_batches)
		valid_acc = valid_acc / len(dev_batches)
		
		end_time = time.time()
		epoch_secs = end_time - start_time
		
		print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_secs}s')
		print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
		print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
		
			
		if valid_acc - best_valid_acc < 0.0001:
			break
		
		if valid_acc > best_valid_acc:
			best_valid_acc = valid_acc
			torch.save(model.state_dict(), config.model_file)

	# 所有的错误 (label, pred)
	err_words_all = []
	test_loss = 0
	test_acc = 0
	for batch in test_batches:
		loss, acc, err_words = evaluate(model, batch, criterion, index_to_word)
		test_loss += loss
		test_acc += acc
		err_words_all.extend(err_words)

	test_loss = test_loss / len(test_batches)
	test_acc = test_acc / len(test_batches)
	print(f'Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')

	# 最常见的topk错误
	err_top = get_top_error_words(err_words_all, config.err_top_num)

	with open(config.err_top_file, "w+") as fw:
		fw.write("\n".join(["{}->{}:{}".format(x[0], x[1], y) for x,y in err_top]))
	
	with open(config.err_file, "w+") as fw:
		fw.write("\n".join(["{}->{}".format(x, y) for x,y in err_words_all]))
	
		
	
if __name__ == '__main__':
	run()

