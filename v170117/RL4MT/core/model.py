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
			

class ActorCriticModel(Model):

	def __init__(self, src_vocab_size, trg_vocab_size, src_embedding_dim, trg_embedding_dim, src_hidden_dim, trg_hidden_dim, attn_dim, src_max_len, trg_max_len, buffer_size, n_max_out, nullsym, discount, name):
		super(ActorCriticModel, self).__init__()
		self.name = name
		self.src_vocab_size = src_vocab_size
		self.trg_vocab_size = trg_vocab_size
		self.src_embedding_dim = src_embedding_dim
		self.trg_embedding_dim = trg_embedding_dim
		self.src_hidden_dim = src_hidden_dim
		self.trg_hidden_dim = trg_hidden_dim
		self.attn_dim = attn_dim
		self.buffer_size = buffer_size
		self.trg_max_len = trg_max_len
		self.nullsym = nullsym # integer
		self.discount = discount

		self.encoder = BiGruEncoder(self.src_vocab_size, self.src_embedding_dim, self.src_hidden_dim, name=self.name+'_encoder')
		self.params += self.encoder.params

		self.actor = Actor(self.buffer_size, self.trg_vocab_size, self.trg_embedding_dim,
						self.src_hidden_dim, self.trg_hidden_dim, self.attn_dim,
						n_max_out, name=self.name+'_actor')
		self.params += self.actor.params

		self.critic = Critic(src_max_len, self.src_hidden_dim, self.trg_hidden_dim, name=self.name+'_critic')
		self.params += self.critic.params


	def ActorPretrainPhase(self, batch_size):

		# inputs placeholder
		self.x = TT.matrix('x', dtype='int64') # (sent_len, batch_size)
		self.x_mask = TT.matrix('x_mask', dtype='float32') # (sent_len, batch_size)
		self.y_reorder = TT.matrix('y_reorder', dtype='int64') # (sent_len, batch_size)
		self.buffer_indices = TT.tensor3('buffer_indices', dtype='int64') # (sent_len, batch_size, buffer_size)
		self.positions = TT.tensor3('positions', dtype='float32') # (sent_len, batch_size, buffer_size)
		self.inputs = [self.x, self.x_mask, self.y_reorder, self.buffer_indices, self.positions]

		# build computation graph

		# encode source sentence
		self.src_context = self.encoder(self.x, self.x_mask)

		# initialize actor state and buffer
		actor_init_state = self.src_context[0,:,self.src_hidden_dim:]
		# actor_init_buffer = TT.as_tensor_variable([[self.nullsym] * self.actor.buffer_size] * batch_size)
		# self.actor.init_sentence(actor_init_buffer, actor_init_state, self.src_context, self.x_mask, freeze_pos_policy=True)
		self.actor.init_sentence(actor_init_state, self.src_context, self.x_mask, freeze_pos_policy=True)

		# pretrain word policy
		scan_sequences = [self.y_reorder, self.buffer_indices, self.positions]
		scan_nonseq = [TT.as_tensor_variable(self.trg_vocab_size)]
		# init_emb = TT.unbroadcast(TT.unbroadcast(self.actor.buffer.dimshuffle('x',0,1), 1), 0)
		# init_state = TT.unbroadcast(TT.unbroadcast(self.actor.state.dimshuffle('x',0,1), 1), 0)
		# init_cost = TT.unbroadcast(TT.zeros(1, dtype="float32"), 0)
		# init_emb = TT.unbroadcast(TT.unbroadcast(self.actor.buffer, 1), 0)
		init_state = TT.unbroadcast(TT.unbroadcast(self.actor.state, 1), 0)
		init_cost = TT.zeros(1, dtype="float32")

		#          |     sequences      |  prev info  | non-sequences |
		def scan_fn(y, buffer_index, pos, state, costs, vocab_size):
			# update buffer and state
			self.actor.state = state
			# compute word probabilities and cost
			probs = self.actor.action_word(pos) # (batch_size, vocab_size)
			y_index = TT.arange(batch_size)*vocab_size + y.flatten()
			cost = -TT.log(probs.flatten()[y_index])
			# generate buffer and state for next step
			new_state = self.actor.state_updater(buffer_index, state)
			return new_state, TT.cast(cost.sum(),'float32').dimshuffle('x')

		results, updates = theano.scan(scan_fn,
					sequences=scan_sequences,
					outputs_info=[init_state, init_cost],
					non_sequences=scan_nonseq)

		# supervision: model cost
		self.costs = results[1].flatten().sum()


	def actor_pretrain_sequence(self, y):
		""" bridging data and model input """
		batch_size = y.shape[0]
		sent_len = y.shape[1]

		# generate random positions and decode order
		position_indices = [] # (batch_size, sent_len)
		orders = [] # (batch_size, sent_len)
		y_reorder = np.zeros((sent_len, batch_size), dtype='int64') # should be length major

		for i in range(batch_size):
			all_pos = range(self.buffer_size) # list, retaining order
			position = random.sample(all_pos, sent_len)
			position.sort() # list, sorted (random insert nullsym)
			position_indices.append(position)
			# pos_set = set(position) # set, without order, faster query
			# null_pos = [x for x in all_pos if x not in pos_set] # list, retaining order
			# positions.append( position + null_pos ) # first decode word, then nullsym

			order = range(sent_len)
			random.shuffle(order)
			# order += range(sent_len, self.buffer_size)
			orders.append(order)

		buffer_ids = []
		buffer_idx = np.array([[self.nullsym] * self.buffer_size] * batch_size) # (batch, sent_len)
		one_hot_len_major_pos = []
		for i in range(sent_len):
			step_position = [position_indices[j][orders[j][i]] for j in range(batch_size)] # i-th step
			for j in range(batch_size):
				y_reorder[i,j] = y[ j, orders[j][i] ]
				buffer_idx[ j, step_position[j] ] = y[ j, orders[j][i] ] # j-th batch, step-position of j-th batch
			buffer_ids.append(np.copy(buffer_idx)) # make a deep copy
			one_hot_step_position = one_hot(step_position, self.buffer_size)
			one_hot_len_major_pos.append(one_hot_step_position)

		positions = np.asarray(one_hot_len_major_pos, dtype='float32')
		buffer_indices = np.asarray(buffer_ids, dtype='int64')

		return y_reorder, buffer_indices, positions


	def CriticPretrainPhase(self, batch_size):

		# inputs placeholder
		self.x = TT.matrix('x', dtype='int64') # (sent_len, batch_size)
		self.x_mask = TT.matrix('x_mask', dtype='float32') # (sent_len, batch_size)
		self.buffer_indices = TT.tensor3('buffer_indices', dtype='int64') # (sent_len, batch_size, buffer_size)
		self.rewards = TT.matrix('rewards', dtype='float32') # (sent_len, batch_size)
		self.inputs = [self.x, self.x_mask, self.buffer_indices, self.rewards]

		# build computation graph

		# encode source sentence
		self.src_context = self.encoder(self.x, self.x_mask)

		# initialize actor state and buffer, and critic
		actor_init_state = self.src_context[0,:,self.src_hidden_dim:]
		self.actor.init_sentence(actor_init_state, self.src_context, self.x_mask,
								freeze_pos_policy=True, freeze_word_policy=True)
		self.critic.init_sentence(self.src_context)

		# pretrain critic
		scan_sequences = [self.buffer_indices, self.rewards]
		scan_nonseq = [TT.as_tensor_variable(self.discount)]
		# init_emb = TT.unbroadcast(TT.unbroadcast(self.actor.buffer, 1), 0)
		init_state = TT.unbroadcast(TT.unbroadcast(self.actor.state, 1), 0)
		init_val = TT.unbroadcast(self.critic(self.actor.state), 0)
		init_trg_val = TT.unbroadcast(TT.zeros(batch_size, dtype="float32"), 0)
		# Note, that target_values is offset by -1, therefore init_trg_val makes no sense

		#          |     sequences      |    previous info   | non-sequences |
		def scan_fn(buffer_index, reward, state, val, trg_val, discount):
			# generate buffer and state for next step
			new_state = self.actor.state_updater(buffer_index, state)
			# compute new value
			new_val = self.critic(state)
			new_trg_val = reward + discount * new_val
			return new_state, new_val, new_trg_val

		results, updates = theano.scan(scan_fn,
					sequences=scan_sequences,
					outputs_info=[init_state, init_val, init_trg_val],
					non_sequences=[self.discount])

		self.values = results[1][:-1]
		self.target_values = results[2][1:]
		self.costs = ((self.target_values - self.values)**2).flatten().sum()


	def critic_pretrain_sequence(self, y):
		""" bridging data and model input """
		batch_size = y.shape[0]
		sent_len = y.shape[1]

		# generate random positions and decode order
		position_indices = [] # (batch_size, sent_len)
		orders = []

		for i in range(batch_size):
			all_pos = range(self.buffer_size) # list, retaining order
			position = random.sample(all_pos, sent_len)
			position.sort() # list, sorted (random insert nullsym)
			position_indices.append(position)
			# pos_set = set(position) # set, without order, faster query
			# null_pos = [x for x in all_pos if x not in pos_set] # list, retaining order
			# positions.append( position + null_pos ) # first decode word, then nullsym

			order = range(sent_len)
			random.shuffle(order)
			# order += range(sent_len, self.buffer_size)
			orders.append(order)

		buffer_ids = [] # (step, batch, sent_len)
		buffer_idx = np.array([[self.nullsym] * self.buffer_size] * batch_size) # (batch, sent_len)
		for i in range(sent_len):
			step_position = [position_indices[j][orders[j][i]] for j in range(batch_size)] # i-th step
			for j in range(batch_size):
				buffer_idx[ j, step_position[j] ] = y[ j, orders[j][i] ] # j-th batch, step-position of j-th batch
			buffer_ids.append(np.copy(buffer_idx)) # make a deep copy

		buffer_indices = np.asarray(buffer_ids, dtype='int64')

		# calculate rewards

		bleus = [[0.] * batch_size] # (step, batch)
		# remove nullsym
		references = y.tolist() # (batch, sent_len)
		ref = [ [w for w in reference if not w == self.nullsym] for reference in references ]
		hypotheses = buffer_indices.tolist() # (step, batch, buffer_size)
		for hypothesis in hypotheses:
			bleu = []
			for j in range(len(hypothesis)):
				hyp = [w for w in hypothesis[j] if not w == self.nullsym]
				bleu.append( sentence_bleu([ ref[j] ], hyp, smoothing_function=smooth_fn2) )
			bleus.append(bleu)
		rewards = [[ bleus[i+1][j] - bleus[i][j] for j in range(batch_size) ] for i in range(sent_len)]

		return buffer_indices, np.asarray(rewards, dtype='float32')


	def ActorCriticTrainingPhase(self, batch_size):

		""" Sampling Graph """
		# Don't share nodes. Might invoke missing input error.
		self.rng = RandomStreams(np.random.randint(int(10e6)))

		# inputs placeholder
		self.x_sampl = TT.matrix('x_sampl', dtype='int64') # (sent_len, batch_size)
		self.x_mask_sampl = TT.matrix('x_mask_sampl', dtype='float32') # (sent_len, batch_size)
		self.max_step = TT.scalar('max_step', dtype='int64')
		self.sampl_inputs = [self.x_sampl, self.x_mask_sampl, self.max_step]

		# encode source sentence
		self.src_context_sampl = self.encoder(self.x_sampl, self.x_mask_sampl)

		# initialize: actor state and buffer
		actor_init_state_sampl = self.src_context_sampl[0,:,self.src_hidden_dim:]
		self.actor.init_sentence(actor_init_state_sampl, self.src_context_sampl, self.x_mask_sampl)
		actor_init_buffer_sampl = TT.as_tensor_variable([[self.nullsym] * self.actor.buffer_size] * batch_size)

		# initial value for scan
		init_buffer_sampl = TT.unbroadcast(TT.unbroadcast(actor_init_buffer_sampl, 1), 0)
		init_state_sampl = TT.unbroadcast(TT.unbroadcast(self.actor.state, 1), 0)
		init_one_hot_pos_sampl = TT.unbroadcast(TT.unbroadcast(TT.zeros((batch_size, self.buffer_size), dtype='float32'), 1), 0)
		init_one_hot_word_sampl = TT.unbroadcast(TT.unbroadcast(TT.zeros((batch_size, self.trg_vocab_size), dtype='float32'), 1), 0)

		#                |                 previous info                          |
		def sampl_scan_fn(buffer_index, state, prev_one_hot_pos, prev_one_hot_word):
			# update state
			self.actor.state = state
			# sample positions
			probs_pos = self.actor.action_pos()
			one_hot_pos = self.rng.multinomial(pvals=probs_pos, dtype='float32')
			# sample words
			probs_word = self.actor.action_word(one_hot_pos)
			one_hot_word = self.rng.multinomial(pvals=probs_word, dtype='float32')
			word_ids = one_hot_word.argmax(axis=-1)
			# make new buffer
			new_buffer = one_hot_pos * word_ids.dimshuffle(0,'x') + (1.-one_hot_pos) * TT.cast(buffer_index, 'float32')
			### Something really WEIRD: if I remove dimshuffle, it gives missing input error ###
			new_buffer = TT.cast(new_buffer, 'int64')
			# generate new state
			new_state = self.actor.state_updater(new_buffer, state)

			return new_buffer, new_state, one_hot_pos, one_hot_word

		sampl_results, sampl_updates = theano.scan(sampl_scan_fn,
												sequences=None,
												outputs_info=[init_buffer_sampl, init_state_sampl, init_one_hot_pos_sampl, init_one_hot_word_sampl],
												n_steps=self.max_step)

		self.buffer_ids = sampl_results[0][1:]
		self.one_hot_positions = sampl_results[2][1:]
		self.one_hot_words = sampl_results[3][1:]

		self.sample = theano.function(inputs=self.sampl_inputs,
									outputs=[self.buffer_ids, self.one_hot_positions, self.one_hot_words],
									updates=sampl_updates) # updates from scan (because of random number)

		""" Decode Graph """

		# inputs placeholder
		self.x = TT.matrix('x', dtype='int64') # (sent_len, batch_size)
		self.x_mask = TT.matrix('x_mask', dtype='float32') # (sent_len, batch_size)		
		self.buffer_indices = TT.tensor3('buffer_indices', dtype='int64') # (max_step, batch_size, buffer_size)
		self.rewards = TT.matrix('rewards', dtype='float32') # (max_step, batch_size)
		self.positions = TT.tensor3('positions', dtype='float32') # (batch_size, buffer_size)
		self.words = TT.tensor3('words', dtype='float32') # (batch_size, vocab_size)
		self.dec_inputs = [self.x, self.x_mask, self.buffer_indices, self.rewards, self.positions, self.words]

		# encode source sentence
		self.src_context = self.encoder(self.x, self.x_mask)

		# initialize: actor state and buffer
		actor_init_state = self.src_context[0,:,self.src_hidden_dim:]
		self.actor.init_sentence(actor_init_state, self.src_context, self.x_mask)

		# initialize: critic
		self.critic.init_sentence(self.src_context)

		# initial values for scan
		init_state = TT.unbroadcast(TT.unbroadcast(self.actor.state, 1), 0)
		init_nl_prob = TT.unbroadcast(TT.zeros(batch_size, dtype='float32'), 0)
		init_value = TT.unbroadcast(self.critic(self.actor.state), 0)
		init_target_value = TT.unbroadcast(TT.zeros(batch_size, dtype="float32"), 0)
		# Note, that target_values is offset by -1, therefore init_trg_val makes no sense

		# decode probabilites and values
		padded_rewards = TT.concatenate([TT.zeros((batch_size,1), dtype='float32'), self.rewards], axis=0)
		dec_scan_sequences = [self.rewards, self.buffer_indices, self.positions, self.words]
		dec_scan_nonseq = [self.discount]
		#              |              sequences                        |        previous info        | non-seq |
		def dec_scan_fn(reward, buffer_index, one_hot_pos, one_hot_word, state, nl_prob, val, trg_val, discount):
			# update state
			self.actor.state = state
			# positions probabilities
			probs_pos = (self.actor.action_pos() * one_hot_pos).sum(axis=-1)
			# words probabilities
			probs_word = (self.actor.action_word(one_hot_pos) * one_hot_word).sum(axis=-1)
			# negative log probabilities
			new_nl_prob = (probs_pos * probs_word).flatten()
			# generate new state
			new_state = self.actor.state_updater(buffer_index, state)
			# new value
			new_val = self.critic(new_state)
			new_trg_val = reward + discount * new_val

			return new_state, new_nl_prob, new_val, new_trg_val

		dec_results, dec_updates = theano.scan(dec_scan_fn,
											sequences=dec_scan_sequences,
											outputs_info=[init_state, init_nl_prob, init_value, init_target_value],
											non_sequences=[self.discount])

		self.nl_probabilities = dec_results[1][1:]
		self.values = dec_results[2]
		self.target_values = dec_results[3][1:]

		self.actor_costs = -TT.log(self.nl_probabilities * self.values[1:]).flatten().sum()
		self.critic_costs = ((self.target_values - self.values[:-1])**2).flatten().sum()


	def get_rewards(self, y, decode_buffers):

		batch_size = y.shape[0]
		sent_len = y.shape[1]
		bleus = [[0.] * batch_size] # (step, batch)
		# remove nullsym
		references = y.tolist() # (batch, sent_len)
		ref = [ [w for w in reference if not w == self.nullsym] for reference in references ]
		hypotheses = decode_buffers.tolist() # (step, batch, buffer_size)
		for hypothesis in hypotheses:
			bleu = []
			for j in range(len(hypothesis)):
				hyp = [w for w in hypothesis[j] if not w == self.nullsym]
				bleu.append( sentence_bleu([ ref[j] ], hyp, smoothing_function=smooth_fn2) )
			bleus.append(bleu)
		rewards = [[ bleus[i+1][j] - bleus[i][j] for j in range(batch_size) ] for i in range(sent_len)]

		return np.asarray(rewards, dtype='float32')


	def TestPhase(self, beam_size=5, explore_ratio=5):
		""" decompose computation graph into multiple theano functions """

		self.x = TT.matrix('x', dtype='int64') #(sent_len, batch_size)
		self.encode = theano.function(inputs=[self.x],
									outputs=[self.encoder(self.x)])

		# actor.state, actor.ctx_p.weighted_src_context, actor.ctx_w.weighted_src_context
		# critic.src_fw_state, critic.src_bw_state
		self.src_context = TT.tensor3('src_context', dtype='float32') #(sent_len, batch_size, 2*src_hidden_dim)
		self.get_init_variables = theano.function(inputs=[self.src_context],
									outputs=[self.actor.W_init(self.src_context[0,:,self.src_hidden_dim:]),
											self.actor.ctx_p.U_src( self.src_context.reshape( (self.src_context.shape[0]*self.src_context.shape[1], 2*self.src_hidden_dim) ) ).reshape( (self.src_context.shape[0], self.src_context.shape[1], self.attn_dim), ndim=3 ),
											self.actor.ctx_w.U_src( self.src_context.reshape( (self.src_context.shape[0]*self.src_context.shape[1], 2*self.src_hidden_dim) ) ).reshape( (self.src_context.shape[0], self.src_context.shape[1], self.attn_dim), ndim=3 ),
											self.critic.W_src_fw(self.src_context[-1,:,:self.src_hidden_dim]),
											self.critic.W_src_bw(self.src_context[0,:,self.src_hidden_dim:]) ])

		self.actor_state = TT.matrix('actor_state', dtype='float32')
		self.actor.state = self.actor_state
		self.actor.ctx_p.src_context = self.src_context
		self.actor.ctx_w.src_context = self.src_context
		self.actor_ctx_p_weighted_src_context = TT.tensor3('actor_ctx_p_weighted_src_context', dtype='float32')
		self.actor.ctx_p.weighted_src_context = self.actor_ctx_p_weighted_src_context
		self.actor_ctx_w_weighted_src_context = TT.tensor3('actor_ctx_w_weighted_src_context', dtype='float32')
		self.actor.ctx_w.weighted_src_context = self.actor_ctx_w_weighted_src_context

		self.pos_act_prob = theano.function(inputs=[self.actor_state, self.src_context, self.actor_ctx_p_weighted_src_context], outputs=[self.actor.action_pos()])

		self.one_hot_position = TT.matrix('one_hot_position', dtype='float32') #(batch_size,buffer_size)
		self.word_act_prob = theano.function(inputs=[self.one_hot_position, self.actor_state, self.src_context, self.actor_ctx_w_weighted_src_context],
									outputs=[self.actor.action_word(self.one_hot_position)])

		self.buffer_indices = TT.matrix('buffer_indices', dtype='int64')
		self.prev_state = TT.matrix('prev_state', dtype='float32')
		self.update_state = theano.function(inputs=[self.buffer_indices, self.prev_state],
									outputs=[self.actor.state_updater(self.buffer_indices, self.prev_state)] )

		self.critic_src_fw_state = TT.matrix('critic_src_fw_state', dtype='float32')
		self.critic.src_fw_state = self.critic_src_fw_state
		self.critic_src_bw_state = TT.matrix('critic_src_bw_state', dtype='float32')
		self.critic.src_bw_state = self.critic_src_bw_state
		self.get_values = theano.function(inputs=[self.actor_state, self.critic_src_fw_state, self.critic_src_bw_state],
									outputs=[self.critic(self.actor_state)])


	def init_variables(self, src_context):

		self.actor.state, self.actor.ctx_p.weighted_src_context, self.actor.ctx_w.weighted_src_context, \
		self.critic.src_fw_state, self.critic.src_bw_state = self.get_init_variables(src_context)

		# self.actor.ctx_p.sent_len = self.actor.ctx_w.sent_len = src_context.shape[0]
		# self.actor.ctx_p.batch_size = self.actor.ctx_w.batch_size = src_context.shape[1]
		self.actor.ctx_p.src_context = self.actor.ctx_w.src_context = src_context
		self.actor.x_mask = None


	def translate(self, x, max_step=200, beam_size=5, explore_ratio=5):

		# encode source sentence
		src_context = self.encode(x)[0]

		# initialize
		self.init_variables(src_context)
		# log.write("src_context:{}\nself.actor.ctx_p.weighted_src_context:{}\nself.actor.ctx_p.sent_len:{}\nself.actor.ctx_p.batch_size:{}\n".format(src_context,self.actor.ctx_p.weighted_src_context.shape,self.actor.ctx_p.sent_len,self.actor.ctx_p.batch_size))

		candidate_results = np.asarray([[self.nullsym] * self.buffer_size], dtype='int64')
		candidate_values = self.get_values(self.actor.state, self.critic.src_fw_state, self.critic.src_bw_state)[0]
		candidate_returns = np.zeros(1, dtype='float32')
		candidate_states = self.actor.state

		completed_results = []
		completed_returns = []

		candidate_num = 1

		step = 0
		while beam_size > 0 and step < max_step:
			if candidate_num <= 0:
				break
			step += 1

			self.actor.state = candidate_states
			candidate_num = candidate_results.shape[0]

			# replicate src_context by candidate_num
			src_context_repl = np.repeat(src_context, candidate_num, axis=1)
			ctx_p_weighted_src_context_repl = np.repeat(self.actor.ctx_p.weighted_src_context, candidate_num, axis=1)

			# select position
			probs_pos = self.pos_act_prob(self.actor.state, src_context_repl, ctx_p_weighted_src_context_repl)[0] # beam as batch
			max_probs_pos = np.argsort(probs_pos, axis=1)[:,:explore_ratio] # (candidate_num, explore_ratio) candidates

			# make one-hot positions: candidate_num*explore_ratio
			max_probs_pos_reshape = max_probs_pos.reshape((candidate_num*explore_ratio, 1))
			one_hot_pos = one_hot( max_probs_pos_reshape.tolist(), self.buffer_size ) # (candidate_num*explore_ratio, buffer_size)

			# replicate states and renew actor state
			self.actor.state = np.repeat(self.actor.state, explore_ratio, axis=1).reshape((candidate_num*explore_ratio, self.trg_hidden_dim))

			# replicate source contexts and weighted source contexts
			src_context_repl = np.repeat(src_context, candidate_num*explore_ratio, axis=1) # (sent_len, batch, 2*hidden_dim)
			ctx_w_weighted_src_context_repl = np.repeat(self.actor.ctx_w.weighted_src_context, candidate_num*explore_ratio, axis=1) # (sent_len, batch, attn_dim)

			# select word
			probs_words = self.word_act_prob(one_hot_pos, self.actor.state, src_context_repl, ctx_w_weighted_src_context_repl)[0]
			max_probs_word = np.argsort(probs_words, axis=1)[:,:explore_ratio] # (candidate_num*explore_ratio, explore_ratio) candidates

			# update state
			# replicate buffer
			buffer_repl = np.repeat(candidate_results, explore_ratio*explore_ratio, axis=1).reshape((candidate_num*explore_ratio*explore_ratio, self.buffer_size))
			# flatten word indices
			new_word_indices = max_probs_word.reshape((candidate_num*explore_ratio*explore_ratio, 1))
			# replicate position indices
			new_pos_indices = np.repeat(max_probs_pos_reshape, explore_ratio, axis=1).reshape((candidate_num*explore_ratio*explore_ratio, 1))
			# renew buffer
			for i in range(candidate_num*explore_ratio*explore_ratio):
				buffer_repl[i][new_pos_indices[i]] = new_word_indices[i]

			# replicate states and renew actor state
			self.actor.state = np.repeat(self.actor.state, explore_ratio, axis=1).reshape((candidate_num*explore_ratio*explore_ratio, self.trg_hidden_dim))
			# update state
			new_states = self.update_state(buffer_repl, self.actor.state)[0]

			# replicate source context states
			self.critic.src_fw_state = np.repeat(self.critic.src_fw_state, candidate_num*explore_ratio*explore_ratio, axis=0)
			self.critic.src_bw_state = np.repeat(self.critic.src_bw_state, candidate_num*explore_ratio*explore_ratio, axis=0)
			# compute value
			new_values = self.get_values(new_states, self.critic.src_fw_state, self.critic.src_bw_state)[0]
			# select top values to remain in beam
			new_candidate_indices = np.argsort(new_values)[:beam_size].tolist()

			new_candidate_results = []
			new_candidate_returns = []
			new_candidate_values = []
			new_candidate_states = []

			for idx in new_candidate_indices:
				cand_idx = idx / (explore_ratio*explore_ratio)
				pos_idx = max_probs_pos_reshape[idx/explore_ratio].flatten()
				word_idx = new_word_indices[idx].flatten()

				if new_values[idx] < candidate_values[cand_idx]:
					completed_results.append(candidate_results[cand_idx])
					completed_returns.append(candidate_returns[cand_idx])
					beam_size -= 1
					continue

				cand_result = candidate_results[cand_idx,:]
				cand_result[pos_idx] = word_idx
				new_candidate_results.append( cand_result.reshape((1,cand_result.shape[0])) )

				new_candidate_values.append(new_values[idx])

				# r_{i-1} = Q_{i-1} - \lambda * Q_i
				cand_ret = candidate_returns[cand_idx] + candidate_values[cand_idx] - self.discount*new_values[idx]
				new_candidate_returns.append(cand_ret)

				new_candidate_states.append( new_states[idx].reshape((1,new_states.shape[1])) )

			candidate_num = len(new_candidate_returns)
			if candidate_num > 0:
				candidate_results = np.concatenate(new_candidate_results, axis=0)
				candidate_values = np.array(new_candidate_values, dtype='float32')
				candidate_returns = np.array(new_candidate_returns, dtype='float32')
				candidate_states = np.concatenate(new_candidate_states, axis=0)
		# end while
		
		return completed_results[ np.argmax( completed_returns ) ]
