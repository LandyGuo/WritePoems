#coding=utf-8

import collections
import numpy as np
import tensorflow as tf
import codecs

#-------------------------------数据预处理---------------------------#
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


poetry_file ='data/poetry.txt'

# 诗集
poetrys = []
with codecs.open(poetry_file, "r",'utf-8') as f:
	for line in f:
		# print line
		try:
			title, content = line.strip().split(':')
			content = content.replace(' ','')
			if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
				continue
			if len(content) < 5 or len(content) > 79:
				continue
			content = '[' + content + ']'
			# print content
			poetrys.append(content)
		except Exception as e: 
			print e

# 按诗的字数排序
poetrys = sorted(poetrys,key=lambda line: len(line))
print(u'唐诗总数: ', len(poetrys))

# 统计每个字出现次数
all_words = []
for poetry in poetrys:
	all_words += [word for word in poetry]
counter = collections.Counter(all_words)
# print counter
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
# print count_pairs
words, _ = zip(*count_pairs)
#add empty char
words = words + (" ",) 
# map word to id
# 每个字映射为一个数字ID
word2idmap = dict(zip(words,range(len(words))))
# 把诗转换为向量形式
word2idfunc = lambda word:  word2idmap.get(word,len(words))
peorty_vecs  = [list(map(word2idfunc,peotry)) for peotry in poetrys]



#batch-wise padding:do padding to the same size(sequence length) of each batch
batch_size = 1
n_batch = (len(peorty_vecs)-1) // batch_size
X_data,Y_data = [],[]
for i in range(n_batch):
	cur_vecs = peorty_vecs[i*batch_size:(i+1)*batch_size]
	current_batch_max_length = max(map(len,cur_vecs))
	batch_matrix = np.full((batch_size,current_batch_max_length),word2idfunc(" "),np.int32)
	for j in range(batch_size):
		batch_matrix[j,:len(cur_vecs[j])] = cur_vecs[j]
	x = batch_matrix
	X_data.append(x)
	y = np.copy(x)
	y[:,:-1] = x[:,1:]
	# print x
	# print y
	Y_data.append(y)
	

#build rnn
vocab_size = len(words)+1
#input_size:(batch_size,feature_length)
input_sequences = tf.placeholder(tf.int32,shape=[batch_size,None])
output_sequences = tf.placeholder(tf.int32,shape=[batch_size,None])


def build_rnn(hidden_units=128,layers=2):
	#embeding
	with tf.variable_scope("embedding"):
		embedding = tf.get_variable("embedding",[vocab_size,hidden_units],dtype=tf.float32)
		#input: batch_size * time_step * embedding_feature
		input = tf.nn.embedding_lookup(embedding,input_sequences)

	basic_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units,state_is_tuple=True)
	stack_cell = tf.nn.rnn_cell.MultiRNNCell([basic_cell]*layers)
	_initial_state = stack_cell.zero_state(batch_size,tf.float32)
	outputs, state = tf.nn.dynamic_rnn(stack_cell, input,initial_state=_initial_state,dtype=tf.float32)
	outputs = tf.reshape(outputs, [-1,hidden_units])

	with tf.variable_scope("softmax"):
		softmax_w  =tf.get_variable("softmax_w",[hidden_units,vocab_size])
		softmax_b  =tf.get_variable("softmax_b",[vocab_size])
		logits = tf.matmul(outputs,softmax_w)+softmax_b

	probs = tf.nn.softmax(logits)

	return logits, probs,stack_cell, _initial_state,state

def train(reload=True):
	logits, probs,_,_,_ = build_rnn()

	targets = tf.reshape(output_sequences,[-1])

	loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], 
		[tf.ones_like(targets, dtype=tf.float32)],len(words))
	cost = tf.reduce_mean(loss)

	learning_rate = tf.Variable(0.002, trainable=False)
	tvars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train_op = optimizer.apply_gradients(zip(grads, tvars))

	global_step = 0
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

		if reload:
			module_file = tf.train.latest_checkpoint('.')
			sess = saver.restore(module_file)
			print "reload sess"

		for epoch in range(50):
			print "learning_rate decrease"
			if global_step%80==0:
				sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
			epoch_steps =  len(zip(X_data,Y_data))
			for step,(x,y) in enumerate(zip(X_data,Y_data)):
				global_step = epoch * epoch_steps + step
				_, los  = sess.run([train_op, cost], feed_dict={
					input_sequences:x,
					output_sequences:y,
					})
				print "epoch:%d steps:%d/%d loss:%3f" % (epoch,step,epoch_steps,los)
				if global_step%100==0:
					print "save model"
					saver.save(sess,"peotry",global_step=epoch)


def write_poem():

	def to_word(weights):
		t = np.cumsum(weights)
		s = np.sum(weights)
		sample = int(np.searchsorted(t, np.random.rand(1)*s))
		print "sample:",sample
		print "len Words:",len(words)
		return words[sample]

	#

	logits, probs,stack_cell, _initial_state, last_state = build_rnn()
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
		module_file = tf.train.latest_checkpoint('.')
		print "load:",module_file
		saver.restore(sess,module_file)

		_state = sess.run(stack_cell.zero_state(1,dtype=tf.float32))

		x = np.array([[word2idfunc('[')]])

		prob_, _state = sess.run([probs,last_state],feed_dict={input_sequences:x,_initial_state:_state})

		word = to_word(prob_)

		poem = ''

		import time
		while word != ']':
			poem += word
			x = np.array([[word2idfunc(word)]])
			[probs_, _state] = sess.run([probs, last_state], feed_dict={input_sequences: x, _initial_state: _state})
			word = to_word(probs_)
			# time.sleep(1)

	return poem


def write_head_poem(heads):

	def to_word(weights):
		#注意:以下注释代码实现了按照分布的概率进行采样，也可用在word2vec中		
		# t = np.cumsum(weights)
		# s = np.sum(weights)
		# sample = int(np.searchsorted(t, np.random.rand(1)*s))
		# print "sample:",sample
		# print "len Words:",len(words)
		sample = np.argmax(weights)
		return words[sample]

	logits, probs,stack_cell, _initial_state, last_state = build_rnn()

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
		module_file = tf.train.latest_checkpoint('.')
		print "load:",module_file
		saver.restore(sess,module_file)

		_state = sess.run(stack_cell.zero_state(1,dtype=tf.float32))

		poem = ''
		add_comma = False
		for head in heads:
			x = head
			add_comma =  not add_comma
			while x!="," and x!="。" and x!=']':
				#add current
				poem += x
				x = np.array([[word2idfunc(x)]])
				#generate next based on current
				prob_, _state = sess.run([probs,last_state],feed_dict={input_sequences:x,_initial_state:_state})
				x = to_word(prob_)
			sign = "," if add_comma else "。"
			poem = poem + sign
		return poem






# train(False)
print(write_poem())
# print(write_head_poem(u"一二三四"))

