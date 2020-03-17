
import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

# data
data_path = os.path.join(pwd_path, 'data')
dev_file = os.path.join(data_path, "bobsue.lm.dev.txt")
test_file = os.path.join(data_path, "bobsue.lm.test.txt")
train_file = os.path.join(data_path, "bobsue.lm.train.txt")
prev_dev_file = os.path.join(data_path, "bobsue.prevsent.dev.tsv")
prev_test_file = os.path.join(data_path, "bobsue.prevsent.test.tsv")
prev_train_file = os.path.join(data_path, "bobsue.prevsent.train.tsv")
word_dict_file = os.path.join(data_path, "bobsue.voc.txt")

PAD = '#PAD#'
PAD_INDEX = 0

# output
output_path = os.path.join(pwd_path, 'output')
model_file = os.path.join(output_path, "model")

err_top_num = 35
err_top_file = os.path.join(output_path, "err_top")
err_file = os.path.join(output_path, "err")

# param
batch_size = 32
epoch_num = 20

embedding_dim = 100
hidden_dim = 200
num_layers = 1
bidirectional = False
dropout = 0.2

learning_rate = 0.001
learning_rate_gamma = 0.5

# 模型结果中 pad位置填充的值
padding_value = -9999999


