import sys
import os.path
import numpy as np
import theano
import theano.tensor as TT
import random
import copy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from RL4MT.core.utils import *
from RL4MT.core.modules import *

smooth_fn2 = SmoothingFunction().method2

class Model(object):

	def __init__(self, name=None):
		super(Model, self).__init__()
		self.name = name
		self.params = []

	def save(self, path):
		values = {}
		for p in self.params:
			values[p.name] = p.get_value()
		np.savez(path, **values)

	def load(self, path):
		if not os.path.exists(path):
			return
		try:
			values = np.load(path)
			for p in self.params:
				if p.name in values:
					if values[p.name].shape != p.get_value().shape:
						raise IncompatibleParameterShapeError(p.name,p.get_value().shape,values[p.name].shape)
					else:
						p.set_value(values[p.name])
						# print("Loaded parameter {}, shape {} .\n".format( p.name,values[p.name].shape ))
				else:
					raise UndefinedParameterError(p.name)
		except UndefinedParameterError, e:
			print e.msg
			sys.exit(1)
		except IncompatibleParameterShapeError, e:
			print e.msg
			sys.exit(1)


class DeepQModel(Model):

	def __init__(self, src_vocab_size, trg_vocab_size, src_embedding_dim, trg_embedding_dim, src_hidden_dim, trg_hidden_dim, attn_dim, max_out_dim, n_max_out, discount, name):
		super(DeepQModel, self).__init__()
		self.name = name
		self.src_vocab_size = src_vocab_size
		self.trg_vocab_size = trg_vocab_size
		self.src_embedding_dim = src_embedding_dim
		self.trg_embedding_dim = trg_embedding_dim
		self.src_hidden_dim = src_hidden_dim
		self.trg_hidden_dim = trg_hidden_dim
		self.attn_dim = attn_dim
		self.max_out_dim = max_out_dim
		self.discount = discount

		self.encoder = BiGruEncoder(self.src_vocab_size, self.src_embedding_dim, self.src_hidden_dim, name=self.name+'_encoder')
		self.params += self.encoder.params

		self.state_updater = StateUpdater(self.trg_vocab_size, self.src_embedding_dim,
									self.src_hidden_dim, self.trg_hidden_dim, self.attn_dim,
									name=self.name+'_state_updater')
		self.params += self.state_updater.params

		self.evaluator = QValue(self.trg_vocab_size, self.trg_embedding_dim,
									self.src_hidden_dim, self.trg_hidden_dim,
									self.max_out_dim, n_max_out, name=self.name+'_evaluator')
		self.params += self.evaluator.params


	def SupervisedPhase(self, batch_size):

		# inputs placeholder
		self.x = TT.matrix('x', dtype='int64') # (sent_len, batch_size)
		self.x_mask = TT.matrix('x_mask', dtype='float32') # (sent_len, batch_size)
		self.y = TT.matrix('y', dtype='int64') # sampled y (sent_len, batch_size)
		self.y_mask = TT.matrix('y_mask', dtype='float32') # sampled y (sent_len, batch_size)
		self.inputs = [self.x, self.x_mask, self.y, self.y_mask]

		# build computation graph

		# encode source sentence
		self.src_context = self.encoder(self.x, self.x_mask)

		# initialize state updater
		self.state_updater.context.get_weighted_src_context(self.src_context, x_mask=self.x_mask)

		init_state = TT.unbroadcast(TT.unbroadcast(self.state_updater.get_init_state(self.src_context), 1), 0)
		init_prob = TT.unbroadcast(TT.unbroadcast(TT.zeros((batch_size,self.trg_vocab_size), dtype='float32'), 1), 0)

		# mle pretrain
		scan_sequences = [ TT.concatenate([TT.zeros((1,batch_size), dtype='int64'), self.y], axis=0) ]
		#        |seq|    prev info          |
		def scan_fn(y, prev_state, prev_prob):
			# renew state
			new_state, new_emb, new_ctx = self.state_updater(y, prev_state)
			# compute prob
			new_prob = self.evaluator(new_state, new_emb, new_ctx, pretrain=True)
			return new_state, new_prob

		results, updates = theano.scan(scan_fn,
								sequences=scan_sequences,
								outputs_info=[init_state, init_prob])

		probs = results[1][:-1]
		y_ids = TT.arange(self.y.flatten().shape[0])*self.trg_vocab_size + self.y.flatten()
		nl_probs = -TT.log(probs.flatten()[y_ids])
		self.costs = ( nl_probs.reshape((self.y.shape[0],self.y.shape[1])) * self.y_mask ).flatten().sum()


	def ReinforcementPhase(self, batch_size, max_step):

		""" Sampling Graph """
		self.rng = RandomStreams(np.random.randint(int(10e6)))
		# inputs placeholder
		self.x_sampl = TT.matrix('x_sampl', dtype='int64') # (sent_len, batch_size)
		self.x_mask_sampl = TT.matrix('x_mask_sampl', dtype='float32') # (sent_len, batch_size)
		self.greedy_eps = TT.scalar('greedy_eps', dtype='float32')
		self.inputs_sampl = [self.x_sampl, self.x_mask_sampl, self.greedy_eps]

		# build computation graph

		# encode source sentence
		self.src_context_sampl = self.encoder(self.x_sampl, self.x_mask_sampl)

		# initialize state updater
		self.state_updater.context.get_weighted_src_context(self.src_context_sampl, x_mask=self.x_mask_sampl)

		init_y_sampl = TT.unbroadcast(TT.zeros(batch_size, dtype='int64'), 0)		
		init_state_sampl = TT.unbroadcast(TT.unbroadcast(self.state_updater.get_init_state(self.src_context_sampl), 1), 0)
		# init_value_sampl = TT.unbroadcast(TT.zeros(batch_size, dtype='float32'), 0)

		#                 |   prev info     |
		def sampl_scan_fn(prev_y, prev_state):
			# renew state
			new_state, new_emb, new_ctx = self.state_updater(prev_y, prev_state)
			# eps-greedy
			new_y = theano.ifelse.ifelse( TT.lt( self.rng.uniform((), low=0., high=1., dtype='float32'), self.greedy_eps ),
								self.rng.uniform((batch_size,), dtype='int64') % self.trg_vocab_size,
								self.evaluator(new_state, new_emb, new_ctx).argmax(axis=-1) ) # bound failed for int64
			return new_y, new_state

		sampl_results, sampl_updates = theano.scan(sampl_scan_fn,
								outputs_info=[init_y_sampl, init_state_sampl],
								n_steps=max_step)

		self.y_sampl = sampl_results[0] # (max_step, batch_size)
		self.outputs_sampl = [self.y_sampl]

		self.sample = theano.function(inputs=self.inputs_sampl, outputs=self.outputs_sampl, updates=sampl_updates)

		""" Q-update Graph """

		# inputs placeholder
		self.x = TT.matrix('x', dtype='int64') # (sent_len, batch_size)
		self.x_mask = TT.matrix('x_mask', dtype='float32') # (sent_len, batch_size)
		self.y = TT.matrix('y', dtype='int64') # sampled y (sent_len, batch_size)
		self.y_mask = TT.matrix('y_mask', dtype='float32') # sampled y (sent_len, batch_size)
		self.rewards = TT.matrix('rewards', dtype='float32') # (sent_len, batch_size)
		self.inputs = [self.x, self.x_mask, self.y, self.y_mask, self.rewards]

		# build computation graph

		# encode source sentence
		self.src_context = self.encoder(self.x, self.x_mask)

		# initialize state updater
		self.state_updater.context.get_weighted_src_context(self.src_context, x_mask=self.x_mask)

		init_y = TT.unbroadcast(TT.zeros(batch_size, dtype='int64'), 0)
		init_state = TT.unbroadcast(TT.unbroadcast(self.state_updater.get_init_state(self.src_context), 1), 0)
		init_value = TT.unbroadcast(TT.zeros(batch_size, dtype='float32'), 0)

		# q-learning
		scan_sequences = [self.y]
		#        |seq|    prev info          |
		def scan_fn(y, prev_state, prev_q_val):
			# renew state
			new_state, new_emb, new_ctx = self.state_updater(y, prev_state)
			# compute q-value
			q_vals = self.evaluator(new_state, new_emb, new_ctx)
			# max values of next word
			new_q_val = TT.max(q_vals, axis=-1)
			return new_state, new_q_val

		results, updates = theano.scan(scan_fn,
								sequences=scan_sequences,
								outputs_info=[init_state, init_value])
		q_values = results[1][:-1]
		next_q_values = results[1][1:]

		q_target = self.rewards[:-1] + self.discount * next_q_values
		self.costs = ( ((q_target - q_values) ** 2) * self.y_mask[:-1] ).flatten().sum()


	def get_reward(self, y, y_mask, y_sampl):
		batch_size = y.shape[1]
		max_len = y_sampl.shape[0]

		bleus = [ [ sentence_bleu([y[:,j].tolist()], y_sampl[:i,j].tolist(), smoothing_function=smooth_fn2)
					for j in xrange(batch_size) ] for i in xrange(max_len+1) ]

		rewards = np.array([ [ bleus[i+1][j] - bleus[i][j]
					for j in xrange(batch_size) ] for i in xrange(max_len) ], dtype='float32') # (max_len, batch_size)

		return rewards

	def get_y_mask(self, y_sampl):
		y_mask = np.ones_like(y_sampl, dtype='float32') # (sent_len, batch_size)
		for i in xrange(y_sampl.shape[1]):
			for j in xrange(y_sampl.shape[0]):
				if y_sampl[j,i] == 0:
					y_mask[j:,i] = 0.
					break
		return y_mask


	def TestPhase(self):
		""" Greedy decode """

		""" decompose computation graph into multiple theano functions """

		self.x = TT.matrix('x', dtype='int64') #(sent_len, batch_size)
		self.encode = theano.function(inputs=[self.x],
									outputs=[self.encoder(self.x)])

		self.src_context = TT.tensor3('src_context', dtype='float32') #(sent_len, batch_size, 2*src_hidden_dim)

		self.get_init_state = theano.function(inputs=[self.src_context], outputs=[self.state_updater.get_init_state(self.src_context)])
		
		self.get_weighted_src_context = theano.function(inputs=[self.src_context],
								outputs=[self.state_updater.context.get_weighted_src_context(self.src_context, output=True)])

		self.weighted_src_context = TT.tensor3('weighted_src_context', dtype='float32')
		self.prev_state = TT.matrix('prev_state', dtype='float32')
		self.y_indices = TT.vector('y_indices', dtype='int64')
		self.state_updater.context.src_context = self.src_context
		self.state_updater.context.weighted_src_context = self.weighted_src_context
		new_state, new_emb, new_ctx = self.state_updater(self.y_indices, self.prev_state)
		new_value = self.evaluator(new_state, new_emb, new_ctx)
		self.update_state_value = theano.function(inputs=[self.prev_state, self.y_indices, self.src_context, self.weighted_src_context], outputs=[new_state, new_value])


	def translate(self, x):
		# encode source and initialize
		src_context = self.encode(x)[0]
		weighted_src_context = self.get_weighted_src_context(src_context)[0]
		state = self.get_init_state(src_context)[0]
		y_idx = 0

		result = []
		x_len = x.shape[0]
		y_len = 0

		while y_len < 3*x_len:
			y_len += 1

			# update state and value
			state, value = self.update_state_value(state, np.array(y_idx, dtype='int64', ndmin=1), src_context, weighted_src_context)
			# greedy decoding
			y_idx = np.argmax(value)
			if y_idx == 0:
				break
			else:
				result.append(y_idx)

		return result


	# This failed to work due to theano issue #5536
	# def TestPhase(self, trg_eos):
	# 	""" Greedy decode """
	# 	# inputs placeholder
	# 	self.x = TT.matrix('x', dtype='int64') # (sent_len, batch_size)
	# 	self.inputs = [self.x]

	# 	# build computation graph

	# 	max_step = self.x.flatten().shape[0] * 3

	# 	# encode source sentence
	# 	self.src_context = self.encoder(self.x)

	# 	# initialize state updater
	# 	self.state_updater.context.get_weighted_src_context(self.src_context)

	# 	init_y = TT.unbroadcast(TT.zeros((1,), dtype='int64'), 0)		
	# 	init_state = TT.unbroadcast(TT.unbroadcast(self.state_updater.get_init_state(self.src_context), 1), 0)

	# 	#          |    prev info     |non-seq|
	# 	def scan_fn(prev_y, prev_state, eos):
	# 		# renew state
	# 		new_state, new_emb, new_ctx = self.state_updater(prev_y, prev_state)
	# 		# greedy decode
	# 		new_y = self.evaluator(new_state, new_emb, new_ctx).argmax(axis=-1)
	# 		return new_y, new_state, theano.scan_module.until( TT.eq(prev_y.flatten()[0], eos) )

	# 	results, updates = theano.scan(scan_fn,
	# 							outputs_info=[init_y, init_state],
	# 							non_sequences=[trg_eos],
	# 							n_steps=max_step)

	# 	self.y = results[0].flatten()[:-1] # (max_step, batch_size)
	# 	self.outputs = [self.y]

	# 	self.translate_fn = theano.function(inputs=self.inputs, outputs=self.outputs)

	# def translate(self, x):
	# 	return self.translate_fn(x)[0]


	# Not gonna use beam-search
	# def TestPhase(self, beam_size=10):
	#	""" Beam-search decode """
	# 	self.beam_size = beam_size

	# 	""" decompose computation graph into multiple theano functions """

	# 	self.x = TT.matrix('x', dtype='int64') #(sent_len, batch_size)
	# 	self.encode = theano.function(inputs=[self.x],
	# 								outputs=[self.encoder(self.x)])

	# 	self.src_context = TT.tensor3('src_context', dtype='float32') #(sent_len, batch_size, 2*src_hidden_dim)

	# 	self.get_init_state = theano.function(inputs=[self.src_context], outputs=[self.state_updater.get_init_state(self.src_context)])
		
	# 	self.get_weighted_src_context = theano.function(inputs=[self.src_context],
	# 							outputs=[self.state_updater.context.get_weighted_src_context(self.src_context, output=True)])

	# 	self.init_state = TT.matrix('init_state', dtype='float32')
	# 	self.get_init_value = theano.function(inputs=[self.init_state], outputs=[self.evaluator.get_init_value(self.init_state)])

	# 	self.weighted_src_context = TT.tensor3('weighted_src_context', dtype='float32')
	# 	self.prev_state = TT.matrix('prev_state', dtype='float32')
	# 	self.y_indices = TT.vector('y_indices', dtype='int64')
	# 	self.state_updater.context.src_context = self.src_context
	# 	self.state_updater.context.weighted_src_context = self.weighted_src_context
	# 	new_state, new_emb, new_ctx = self.state_updater(self.y_indices, self.prev_state)
	# 	new_value = self.evaluator(new_state, new_emb, new_ctx)
	# 	self.update_state_value = theano.function(inputs=[self.prev_state, self.y_indices, self.src_context, self.weighted_src_context], outputs=[new_state, new_value])


	# def translate(self, x):
	# 	# encode source and initialize
	# 	src_context = self.encode(x)[0]
	# 	weighted_src_context = self.get_weighted_src_context(src_context)[0]
	# 	init_state = self.get_init_state(src_context)[0]
	# 	init_value = self.get_init_value(init_state)[0].flatten()[0]
	# 	init_y = np.zeros(1, dtype='int64')

	# 	candidate_results = [[]]
	# 	candidate_values = [init_value]
	# 	candidate_returns = [0.]
	# 	candidate_states = init_state
	# 	candidate_y = init_y

	# 	completed_results = []
	# 	completed_returns = []

	# 	x_len = x.shape[0]
	# 	y_len = 0
	# 	beam_size = self.beam_size

	# 	while len(candidate_results) > 0 and beam_size > 0 and y_len < 3*x_len:
	# 		y_len += 1
	# 		candidate_num = len(candidate_results)

	# 		src_context_repl = np.repeat(src_context, candidate_num, axis=1) # (sent_len, batch_size, 2*src_hidden_dim)
	# 		weighted_src_context_repl = np.repeat(weighted_src_context, candidate_num, axis=1)

	# 		# update state and value
	# 		new_states, new_values = self.update_state_value(candidate_states, candidate_y, src_context_repl, weighted_src_context_repl)
	# 		# top beam_size action
	# 		best_indices = [ (i/self.trg_vocab_size,i%self.trg_vocab_size)
	# 						for i in np.argsort(new_values.flatten())[:beam_size] ]

	# 		# new candidate info
	# 		new_candidate_y = [ j for (i,j) in best_indices ]
	# 		new_candidate_values = [ new_values[i,j] for (i,j) in best_indices ]
	# 		new_candidate_returns = [ candidate_values[best_indices[k][0]] # q_{i-1}
	# 								- self.discount*new_candidate_values[k] # - \gamma * q_i
	# 								+ candidate_returns[best_indices[k][0]] # + g_{i-1}
	# 								for k in xrange(len(new_candidate_values)) ]
	# 		new_candidate_states = [ new_states[i] for (i,j) in best_indices ]
	# 		new_candidate_results = [ candidate_results[i] + [j] for (i,j) in best_indices ]

	# 		# update candidate
	# 		candidate_results = new_candidate_results
	# 		candidate_values = new_candidate_values
	# 		candidate_returns = new_candidate_returns
	# 		candidate_states = new_candidate_states
	# 		candidate_y = new_candidate_y

	# 		# check eos
	# 		for i in xrange(len(candidate_results)-1,-1,-1):
	# 			if candidate_y[i] == 0:
	# 				completed_results.append(candidate_results[i])
	# 				completed_returns.append(candidate_returns[i])
	# 				del candidate_results[i], candidate_values[i], candidate_returns[i], candidate_states[i], candidate_y[i]
	# 				beam_size -= 1

	# 	# dirty tricks: return results even if eos is not generated
	# 	if len(completed_results) == 0:
	# 		completed_results = candidate_results
	# 		completed_returns = candidate_returns

	# 	return completed_results[ np.argmax( completed_returns ) ]