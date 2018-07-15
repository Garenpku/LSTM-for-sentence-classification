from sklearn.preprocessing import OneHotEncoder
from clean import clean_str
from word2vec import word2vec
from data_helper import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
import re

start_time = time.time()

# Text file containing words for training
training_file = 'clothing.csv'

# Checkpoint
checkpoint_dir = "./train/" + str(int(time.time()))
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Parameters for lstm model
batch_size = 64
hidden_size = 64
label_size = 5  #1, 2, 3, 4, 5
training_length = 10000
learning_rate = 0.1
train_loop = 4000
train_test_ratio = 0.8

# Parameters for skip-gram model
embedding_size = 128
batch_size_skip = 128
window_size = 3
learning_rate_skip = 0.3
num_sampled = 64
skip_loop = 10000


class batch_generator():
    def __init__(self, raw, word2id, label, batch_size):
        self.train_x = [[word2id[word] for word in line.split(' ')] for line in raw]
        self.max_len = max([len(line) for line in self.train_x])
        for i in range(len(self.train_x)):
            for _ in range(len(self.train_x[i]), self.max_len):
                self.train_x[i].append(vocab_size - 1)
        ohe = OneHotEncoder().fit([[1], [2], [3], [4], [5]])
        self.train_y = ohe.transform(label).toarray()
        self.batch_size = batch_size
        self.length = len(raw)
        self.cur_pos = 0
    def next(self):
        if self.cur_pos + self.batch_size < self.length:
            x = self.train_x[self.cur_pos:self.cur_pos + self.batch_size]
            y = self.train_y[self.cur_pos:self.cur_pos + self.batch_size]
            self.cur_pos += self.batch_size
            return np.array(x), np.array(y)
        else:
            record = self.cur_pos
            self.cur_pos = self.batch_size - self.length + self.cur_pos
            if self.cur_pos == 0:
                return np.array(self.train_x[record:self.length]), np.array(self.train_y[record:self.length])
            try:
                return np.concatenate((self.train_x[record:self.length], self.train_x[:self.cur_pos])), np.concatenate((self.train_y[record:self.length], self.train_y[:self.cur_pos]))
            except:
                print(np.array(self.train_x[record:self.length]).shape)
                print(np.array(self.train_x[:self.cur_pos]).shape)
                raise ValueError

 

raw_text, label = data_helper(training_file)
print("Loaded training data...")

#Call word2vec interface to generate word representation
embed, word2id, id2word = word2vec(raw_text,
    batch_size = batch_size_skip, 
    embedding_size = embedding_size, 
    window_size = window_size, 
    loop = skip_loop, 
    learning_rate = learning_rate_skip,
    num_sampled = num_sampled)
print(embed.shape)
embed = np.concatenate((embed, [[0.] * embedding_size])).astype(np.float32)
print(embed.shape)
vocab_size = embed.shape[0]

embedding_matrix = tf.Variable(embed)

print("word2vec building finished.")
#train-test split
split_point = int(len(raw_text) * train_test_ratio)
batch = batch_generator(raw_text[:split_point], word2id, label[:split_point], batch_size)
batch_test = batch_generator(raw_text[split_point:], word2id, label[split_point:], batch_size)

#Input gate
Wi = tf.Variable(tf.truncated_normal([embedding_size, hidden_size], -1, 1))
Ui = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], -1, 1))
bi = tf.Variable(tf.truncated_normal([1, hidden_size], -1, 1))
#Forget gate
Wf = tf.Variable(tf.truncated_normal([embedding_size, hidden_size], -1, 1))
Uf = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], -1, 1))
bf = tf.Variable(tf.truncated_normal([1, hidden_size], -1, 1))
#Output gate
Wo = tf.Variable(tf.truncated_normal([embedding_size, hidden_size], -1, 1))
Uo = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], -1, 1))
bo = tf.Variable(tf.truncated_normal([1, hidden_size], -1, 1))
#New memory cell
Wc = tf.Variable(tf.truncated_normal([embedding_size, hidden_size], -1, 1))
Uc = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], -1, 1))
bc = tf.Variable(tf.truncated_normal([1, hidden_size], -1, 1))
#Softmax parameters
W = tf.Variable(tf.truncated_normal([hidden_size, label_size], -1, 1))
b = tf.Variable(tf.truncated_normal([1, label_size], -1, 1))
#Initial states
saved_c = tf.Variable(tf.zeros([batch_size, hidden_size]), trainable = False)
saved_h = tf.Variable(tf.zeros([batch_size, hidden_size]), trainable = False)

def lstm_cell(x, h, last_cell):
    input_gate = tf.sigmoid(tf.matmul(x, Wi) + tf.matmul(h, Ui) + bi)
    forget_gate = tf.sigmoid(tf.matmul(x, Wf) + tf.matmul(h, Uf) + bf)
    output_gate = tf.sigmoid(tf.matmul(x, Wo) + tf.matmul(h, Uo) + bo)
    new_cell = tf.tanh(tf.matmul(x, Wc) + tf.matmul(h, Uc) + bc)
    final_cell = forget_gate * last_cell + input_gate * new_cell
    final_hidden = output_gate * tf.tanh(last_cell)
    return final_hidden, final_cell

# tf Graph input
x_id = tf.placeholder(tf.int32, [None, batch.max_len])
x = tf.nn.embedding_lookup(embedding_matrix, x_id)
y = tf.placeholder(tf.int32, [None, label_size])

# train LSTM model
for i in range(batch.max_len):
    saved_h, saved_c = lstm_cell(x[:, i], saved_h, saved_c)

#soft-max layer
pred = tf.nn.softmax(tf.matmul(saved_h, W) + b)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis = 1), tf.argmax(y, axis = 1)), dtype = tf.float32))

#optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

#summary
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
merged_summary = tf.summary.merge_all()

#initializer
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
    sess.run(init)
    for i in range(train_loop):
        train_x, train_y = batch.next()
        feed_dict = {x_id : train_x, y : train_y}
        _, los, summary, acc = sess.run([optimizer, loss, merged_summary, accuracy], feed_dict = feed_dict)
        writer.add_summary(summary, i)
        print("training step", i, "loss:", los, "accuracy:", acc)
    acc = 0
    loop_time = int((len(raw_text) - split_point) / batch_size)
    for i in range(loop_time):
        test_x, test_y = batch_test.next()
        feed_dict = {x_id : train_x, y : train_y}
        acc_tmp, los = sess.run([accuracy, loss], feed_dict = feed_dict)
        acc += acc_tmp
    print("Final accuracy: ", acc / loop_time)



