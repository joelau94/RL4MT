import numpy as np
import theano
import theano.tensor as TT

from RL4MT.core.utils import *
# set PYTHONPATH to parent directory of RL4MT

class Module(object):

	def __init__(self, name=None):
		self.name = name
		self.params = []


class Linear(Module):

	def __init__(self, input_dim, output_dim, name, use_bias=True):
		super(Linear, self).__init__()
		self.name = name
		self.use_bias = use_bias
		self.W = init_weight((input_dim,output_dim), name=self.name+'_W')
		self.params += [self.W]
		if self.use_bias:
			self.b = init_bias(output_dim, name=name+'_b')
			self.params += [self.b]

	def __call__(self, _input):
		output = TT.dot(_input,self.W)
		if self.use_bias:
			output += self.b
		return output


class EmbeddingLookup(Module):

	def __init__(self, vocab_size, embedding_dim, name, use_bias=True):
		super(EmbeddingLookup, self).__init__()
		self.name = name
		self.W_emb = init_weight((vocab_size, embedding_dim),name+'_W_emb')
		self.use_bias = use_bias
		self.params += [self.W_emb]
		if use_bias:
			self.b = init_bias(embedding_dim,name+'_b')
			self.params += [self.b]

	def __call__(self, index):
		if self.use_bias:
			return self.W_emb[index]+self.b
		else:
			return self.W_emb[index]


class UniGruEncoder(Module):

	def __init__(self, embedding_dim, hidden_dim, name):
		super(UniGruEncoder, self).__init__()
		self.name = name
		self.hidden_dim = hidden_dim

		self.W_hzr = Linear(embedding_dim, 3*hidden_dim, name=self.name+'_W_hzr')
		self.params += self.W_hzr.params

		self.U_zr = Linear(hidden_dim, 2*hidden_dim, name=self.name+'_U_zr', use_bias=False)
		self.params += self.U_zr.params

		self.U_h = Linear(hidden_dim, hidden_dim, name=self.name+'_U_h', use_bias=False)
		self.params += self.U_h.params

	def step(self, weighted_inputs, prev_h, mask=None):

		h_input, z_input, r_input = split(weighted_inputs, 3)

		z_hidden, r_hidden = split(self.U_zr(prev_h), 2)

		z = TT.nnet.sigmoid(z_input + z_hidden)
		r = TT.nnet.sigmoid(r_input + r_hidden)

		h_hidden = self.U_h(r*prev_h)

		proposed_h = TT.tanh(h_input + h_hidden)

		h = (1.-z) * prev_h + z * proposed_h

		if mask is not None:
			mask = mask.dimshuffle(0, 'x')
			return mask * h + (1.-mask) * prev_h
		else:
			return h

	def __call__(self, inputs, sent_len, init_state=None, batch_size=1, mask=None):

		init_state = TT.zeros((batch_size, self.hidden_dim), dtype='float32')
		# init_state = TT.alloc(np.float32(0.), batch_size, self.hidden_dim)

		weighted_inputs = self.W_hzr(inputs).reshape( (sent_len, batch_size, 3*self.hidden_dim) )

		if mask is not None:
			sequences = [weighted_inputs, mask]
			fn = lambda x, m, h : self.step(x, h, mask=m)
		else:
			sequences = [weighted_inputs]
			fn = lambda x, h : self.step(x, h)

		results, updates = theano.scan(fn,
							sequences=sequences,
							outputs_info=[init_state])

		return results


class BiGruEncoder(Module):

	def __init__(self, vocab_size, embedding_dim, hidden_dim, name):
		super(BiGruEncoder, self).__init__()
		self.name = name
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim

		self.embedder = EmbeddingLookup(self.vocab_size, self.embedding_dim, name=self.name+'_embedder')
		self.params += self.embedder.params

		self.forward_gru = UniGruEncoder(self.embedding_dim, self.hidden_dim, self.name+'_forward_gru')
		self.params += self.forward_gru.params

		self.backward_gru = UniGruEncoder(self.embedding_dim, self.hidden_dim, self.name+'_backward_gru')
		self.params += self.backward_gru.params

	def __call__(self, inputs, inputs_mask=None):

		sent_len = inputs.shape[0]
		batch_size = inputs.shape[1]

		embedding = self.embedder(inputs.flatten()) #(sent_len, batch_size, embedding_dim)

		if inputs_mask is not None:
			forward_context = self.forward_gru(embedding,
											sent_len=sent_len,
											batch_size=batch_size,
											mask=inputs_mask) 
			#(sent_len, batch_size, hidden_dim)
			backward_context = self.backward_gru(embedding[::-1],
											sent_len=sent_len,
											batch_size=batch_size,
											mask=inputs_mask[::-1])
		else:
			forward_context = self.forward_gru(embedding,
											sent_len=sent_len,
											batch_size=batch_size) 
			#(sent_len, batch_size, hidden_dim)
			backward_context = self.backward_gru(embedding[::-1],
											sent_len=sent_len,
											batch_size=batch_size)

		context = TT.concatenate([ forward_context, backward_context[::-1] ], axis=2) #(sent_len, batch_size, 2*hidden_dim)

		return context


class Attention(Module):

	def __init__(self, src_hidden_dim, trg_hidden_dim, attn_dim, name):
		super(Attention, self).__init__()
		self.name = name
		self.src_hidden_dim = src_hidden_dim
		self.attn_dim = attn_dim

		self.U_src = Linear(2*self.src_hidden_dim, self.attn_dim, name=self.name+'_U_src')
		self.params += self.U_src.params

		self.W_trg = Linear(trg_hidden_dim, self.attn_dim, name=self.name+'_W_trg', use_bias=False)
		self.params += self.W_trg.params

		self.v = Linear(self.attn_dim, 1, name=self.name+'_v', use_bias=False)
		self.params += self.v.params

		self.src_context = TT.tensor3(dtype='float32')
		self.weighted_src_context = TT.tensor3(dtype='float32')

	def get_weighted_src_context(self, src_context):

		self.sent_len = src_context.shape[0]
		self.batch_size = src_context.shape[1]
		self.src_context = src_context
		# self.weighted_src_context = self.U_src(self.src_context).reshape( (self.sent_len, self.batch_size, self.attn_dim) )
		src_context_flat = self.src_context.reshape( (self.sent_len*self.batch_size, 2*self.src_hidden_dim) )
		self.weighted_src_context = self.U_src(src_context_flat).reshape( (self.sent_len, self.batch_size, self.attn_dim), ndim=3 )
		# (sent_length, batch_size, attn_dim)

	def __call__(self, trg_state, mask=None):

		self.sent_len = self.weighted_src_context.shape[0]
		self.batch_size = self.weighted_src_context.shape[1]
		energy = self.v( self.weighted_src_context + self.W_trg(trg_state) ).reshape( (self.sent_len, self.batch_size) ) # (sent_len, batch_size)
		energy = TT.exp(energy)

		if mask:
			energy *= mask

		normalizer = energy.sum(axis=0).reshape((1,self.batch_size)) # (batch_size,) --> (batch_size,1)
		self.attention_weight = (energy/normalizer).reshape((self.sent_len,self.batch_size,1)) #(sent_len, batch_size, 1)

		attention_state = (self.src_context*self.attention_weight).sum(axis=0) #(batch_size, 2*src_hidden_dim)

		return attention_state


class ActorStateUpdater(Module):

	def __init__(self, max_len, vocab_size, embedding_dim, hidden_dim, name):
		super(ActorStateUpdater, self).__init__()
		self.name = name
		self.max_len = max_len
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim

		self.embedder = EmbeddingLookup(vocab_size,embedding_dim,name=self.name+'_embedder')
		self.params += self.embedder.params

		self.W_hzr = Linear(max_len*embedding_dim, 3*hidden_dim, name=self.name+'_W_hzr')
		self.params += self.W_hzr.params

		self.U_zr = Linear(hidden_dim, 2*hidden_dim, name=self.name+'_U_zr', use_bias=False)
		self.params += self.U_zr.params

		self.U_h = Linear(hidden_dim, hidden_dim, name=self.name+'_U_h', use_bias=False)
		self.params += self.U_h.params

	def __call__(self, new_buffer_index, prev_state):

		batch_size = new_buffer_index.shape[0]

		new_buffer = self.embedder(new_buffer_index).reshape((batch_size, self.max_len * self.embedding_dim))

		weighted_inputs = self.W_hzr(new_buffer) # (batch_size, 3*self.hidden_dim)

		h_input, z_input, r_input = split(weighted_inputs, 3)

		z_hidden, r_hidden = split(self.U_zr(prev_state), 2)

		z = TT.nnet.sigmoid(z_input + z_hidden)
		r = TT.nnet.sigmoid(r_input + r_hidden)

		h_hidden = self.U_h(r*prev_state)

		proposed_h = TT.tanh(h_input + h_hidden)

		new_state = (1.-z) * prev_state + z * proposed_h # (batch_size, hidden_dim)

		return new_state


class Actor(Module):

	def __init__(self, buffer_size, vocab_size, embedding_dim, src_hidden_dim, trg_hidden_dim, attn_dim, n_max_out, name):
		super(Actor, self).__init__()
		self.name = name
		self.buffer_size = buffer_size
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim

		self.state_updater = ActorStateUpdater(self.buffer_size, self.vocab_size, embedding_dim, trg_hidden_dim, name=self.name+'_state_updater')
		self.params += self.state_updater.params

		# for position predictor
		self.ctx_p = Attention(src_hidden_dim, trg_hidden_dim, attn_dim, name=self.name+'_ctx_p')
		self.params += self.ctx_p.params
		self.W_sp = Linear(2*src_hidden_dim, attn_dim, name=self.name+'_W_sp', use_bias=False)
		self.params += self.W_sp.params
		self.W_tp = Linear(trg_hidden_dim, attn_dim, name=self.name+'_W_tp')
		self.params += self.W_tp.params
		self.W_pos = Linear(attn_dim/n_max_out, self.buffer_size, name=self.name+'_W_pos')
		self.params += self.W_pos.params

		# position info for word prediciton
		self.W_th_context = Linear(trg_hidden_dim, self.buffer_size, name=self.name+'_W_th_context')
		self.params += self.W_th_context.params
		self.W_th_position = Linear(trg_hidden_dim, self.buffer_size, name=self.name+'_W_th_position')
		self.params += self.W_th_position.params

		# for word predictor
		self.ctx_w = Attention(src_hidden_dim, trg_hidden_dim, attn_dim, name=self.name+'_ctx_w')
		self.params += self.ctx_w.params
		self.W_sw = Linear(2*src_hidden_dim, attn_dim, name=self.name+'_W_sw', use_bias=False)
		self.params += self.W_sw.params
		self.W_tw = Linear(self.buffer_size, attn_dim, name=self.name+'_W_tw') # input dim should match position vector
		self.params += self.W_tw.params
		self.W_word = Linear(attn_dim/n_max_out, self.vocab_size, name=self.name+'_W_word')
		self.params += self.W_word.params

		self.W_init = Linear(src_hidden_dim, trg_hidden_dim, name=self.name+'_W_init')
		self.params += self.W_init.params

		self.action_word_params = self.state_updater.params + self.W_th_context.params + self.W_th_position.params \
									+ self.ctx_w.params + self.W_sw.params + self.W_tw.params + self.W_word.params \
									+ self.W_init.params

		# self.buffer = TT.matrix(dtype='float32')
		self.state = TT.matrix(dtype='float32')
		self.word_pos_state = TT.matrix(dtype='float32')


	def init_sentence(self, init_state, src_context, src_mask=None, freeze_pos_policy=False, freeze_word_policy=False):
		
		self.state = self.W_init(init_state)

		if not freeze_pos_policy:
			self.ctx_p.get_weighted_src_context(src_context)
		if not freeze_word_policy:
			self.ctx_w.get_weighted_src_context(src_context)
		self.x_mask = src_mask

	def action_pos(self):
		
		if hasattr(self, "x_mask") and self.x_mask is not None:
			position_context = self.ctx_p(self.state, mask=self.x_mask)
		else:
			position_context = self.ctx_p(self.state)
		probs_pos = softmax( self.W_pos( MaxOut( self.W_sp(position_context) + self.W_tp(self.state) ) ) )

		return probs_pos

	def action_word(self, position):
		""" 
		position is one-hot (batch_size,buffer_size)
		self.state (batch_size,trg_hidden_dim) --> (batch_size,buffer_size)
		"""

		self.word_pos_state = (1.-position) * self.W_th_context(self.state) + position * self.W_th_position(self.state)
		if hasattr(self, "x_mask") and self.x_mask is not None:
			word_context = self.ctx_w(self.state, mask=self.x_mask)
		else:
			word_context = self.ctx_w(self.state)
		probs_word = softmax( self.W_word( MaxOut( self.W_sw(word_context) + self.W_tw(self.word_pos_state) ) ) )

		return probs_word


class Critic(Module):

	def __init__(self, src_max_len, src_hidden_dim, trg_hidden_dim, name):
		super(Critic, self).__init__()
		self.name = name
		self.src_max_len = src_max_len
		self.src_hidden_dim = src_hidden_dim

		self.W_src_fw = Linear(self.src_hidden_dim,trg_hidden_dim,name=self.name+'_W_src_fw')
		self.params += self.W_src_fw.params
		self.W_src_bw = Linear(self.src_hidden_dim,trg_hidden_dim,name=self.name+'_W_src_bw')
		self.params += self.W_src_bw.params

		self.W_trg_fw = Linear(trg_hidden_dim,trg_hidden_dim,name=self.name+'_W_trg_fw')
		self.params += self.W_trg_fw.params
		self.W_trg_bw = Linear(trg_hidden_dim,trg_hidden_dim,name=self.name+'_W_trg_bw')
		self.params += self.W_trg_bw.params

		self.src_fw_state = TT.matrix(dtype='float32')
		self.src_bw_state = TT.matrix(dtype='float32')

	def init_sentence(self, src_context, src_mask=None):

		sent_len = src_context.shape[0]
		batch_size = src_context.shape[1]
		if src_mask is not None:
			src_context = src_context * src_mask.dimshuffle(0,1,'x')
		self.src_fw_state = self.W_src_fw(src_context[-1,:,:self.src_hidden_dim]) # (batch_size, trg_hidden_dim)
		self.src_bw_state = self.W_src_bw(src_context[0,:,self.src_hidden_dim:]) # (batch_size, trg_hidden_dim)

	def __call__(self, trg_state):

		value = self.src_fw_state * self.W_trg_fw(trg_state) + self.src_bw_state * self.W_trg_bw(trg_state)
		return TT.tanh(value.sum(axis=1).flatten())
