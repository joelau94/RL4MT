import numpy as np
import cPickle as pkl
import os.path

class Dataset(object):
	""" Dataset for training """
	def __init__(self, data_config):

		self.config = data_config
		self.mode = "batch" # "single"
		self.cursor = 0
		self.iters = 0

		self.source = pkl.load(open(self.config['src_shuf'],'rb'))
		self.target = pkl.load(open(self.config['trg_shuf'],'rb'))
		assert len(self.source) == len(self.target)
		num = 0
		while num < len(self.source):
			if len(self.source[num]) > self.config['src_max_len'] or len(self.target[num]) > self.config['trg_max_len']:
				del self.source[num]
				del self.target[num]
			else:
				num += 1
		self.size = len(self.source)

		self.peeked_batch = None
		self.peeked_batch_cursor = 0

	def next(self):
		self.iters += 1

		if self.mode == "single":
			# for reinforcement phase
			self.peeked_batch = None
			self.peeked_batch_cursor = 0
			x = np.transpose( np.asmatrix( self.source[self.cursor] ))
			y = np.asmatrix( self.target[self.cursor] )
			self.cursor = (self.cursor + 1) % self.size
			return x, y

		elif self.mode == "batch":
			# for supervised phase
			if self.peeked_batch is None or self.peeked_batch_cursor == self.config['peek_num']:
				# Peek peek_num batches. Next peek starts at peek_stop % self.size
				peek_start = self.cursor
				peek_stop = peek_start + self.config['batch_size'] * self.config['peek_num']				

				self.peeked_batch = [ [ self.source[i%self.size], self.target[i%self.size] ]
										for i in range(peek_start, peek_stop) ]
				self.peeked_batch = sorted( self.peeked_batch, key= lambda x : max(len(x[0]),len(x[1])) )
				self.peeked_batch_cursor = 0

			batch_start = self.peeked_batch_cursor * self.config['batch_size']
			batch_stop = batch_start + self.config['batch_size']
			batch_current = np.asarray( self.peeked_batch[batch_start:batch_stop] )

			x_proto = batch_current[:,0]
			y_proto = batch_current[:,1]

			# pad batch
			x_len = max([len(s) for s in x_proto])
			y_len = max([len(s) for s in y_proto])

			# assume eos's index is 0 (to reduce calculation)
			x = np.zeros((x_len, len(x_proto)), dtype='int64') # (sent_len, batch_size)
			y = np.zeros((y_len, len(y_proto)), dtype='int64') # (sent_len, batch_size)

			x_mask = np.zeros((x_len, len(x_proto)), dtype='float32') # (sent_len, batch_size)
			y_mask = np.zeros((y_len, len(y_proto)), dtype='float32') # (sent_len, batch_size)

			for i in range(self.config['batch_size']):
				x[:len(x_proto[i]),i] = x_proto[i] # length major
				x_mask[:len(x_proto[i]),i] = 1.
				y[:len(y_proto[i]),i] = y_proto[i] # length major
				y_mask[:len(y_proto[i]),i] = 1.

			self.cursor = ( self.cursor + self.config['batch_size'] ) % self.size
			self.peeked_batch_cursor += 1
			return x, x_mask, y, y_mask


	def seek(self, cursor):
		self.cursor = cursor

	def set_mode(self, mode):
		self.mode = mode

	def get_cursor(self):
		return self.cursor

	def get_iternum(self):
		return self.iters

	def save_status(self, stat_file):
		status = {}
		status['iters'] = self.get_iternum()
		status['cursor'] = self.get_cursor()
		pkl.dump(status, open(stat_file,'wb'))

	def load_status(self, stat_file):
		if os.path.exists(stat_file):
			status = pkl.load(open(stat_file,'rb'))
			self.cursor = status['cursor']
			self.iters = status['iters']
		else:
			print("Status file not found!")

""" Utility functions for translation """

def sentence_i2w(sentence, i2w_list, eossym):
	return ' '.join([ i2w_list[sentence[i]] for i in xrange(len(sentence)) if not sentence[i] == eossym ])

def sentence_w2i(sentence, w2i_dict, unksym):
	indices = [ w2i_dict[sentence[i]] if w2i_dict.has_key(sentence[i]) else unksym for i in range(len(sentence))]
	return np.asarray(indices, dtype='int64').reshape((len(indices),1))
