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
		self.x_mask = TT.matrix(dtype='float32')

	def get_weighted_src_context(self, src_context, x_mask=None, output=False):

		self.sent_len = src_context.shape[0]
		self.batch_size = src_context.shape[1]
		self.src_context = src_context
		# self.weighted_src_context = self.U_src(self.src_context).reshape( (self.sent_len, self.batch_size, self.attn_dim) )
		src_context_flat = self.src_context.reshape( (self.sent_len*self.batch_size, 2*self.src_hidden_dim) )
		self.weighted_src_context = self.U_src(src_context_flat).reshape( (self.sent_len, self.batch_size, self.attn_dim), ndim=3 )
		# (sent_length, batch_size, attn_dim)
		self.x_mask = x_mask
		if output:
			return self.weighted_src_context

	def __call__(self, trg_state):

		self.sent_len = self.weighted_src_context.shape[0]
		self.batch_size = self.weighted_src_context.shape[1]
		energy = self.v(TT.tanh( self.weighted_src_context + self.W_trg(trg_state) )).reshape( (self.sent_len, self.batch_size) ) # (sent_len, batch_size)
		energy = TT.exp(energy)

		if self.x_mask is not None:
			energy *= self.x_mask

		normalizer = energy.sum(axis=0).reshape((1,self.batch_size)) # (batch_size,) --> (batch_size,1)
		self.attention_weight = (energy/normalizer).reshape((self.sent_len,self.batch_size,1)) #(sent_len, batch_size, 1)

		attention_state = (self.src_context*self.attention_weight).sum(axis=0) #(batch_size, 2*src_hidden_dim)

		return attention_state


class StateUpdater(Module):

	def __init__(self, vocab_size, trg_embedding_dim, src_hidden_dim, trg_hidden_dim, attn_dim, name):
		super(StateUpdater, self).__init__()
		self.name = name
		self.src_hidden_dim = src_hidden_dim

		self.embedder = EmbeddingLookup(vocab_size, trg_embedding_dim, name=self.name+'_embedder')
		self.params += self.embedder.params

		self.context = Attention(src_hidden_dim, trg_hidden_dim, attn_dim, name=self.name+'_context')
		self.params += self.context.params

		self.W_init = Linear(src_hidden_dim, trg_hidden_dim, name=self.name+'_W_init')
		self.params += self.W_init.params

		self.W_hzr = Linear(trg_embedding_dim, 3*trg_hidden_dim, name=self.name+'_W_hzr')
		self.params += self.W_hzr.params

		self.U_zr = Linear(trg_hidden_dim, 2*trg_hidden_dim, name=self.name+'_U_zr', use_bias=False)
		self.params += self.U_zr.params

		self.U_h = Linear(trg_hidden_dim, trg_hidden_dim, name=self.name+'_U_h', use_bias=False)
		self.params += self.U_h.params

		self.C_hzr = Linear(2*src_hidden_dim, 3*trg_hidden_dim, name=self.name+'_C_hzr', use_bias=False)
		self.params += self.C_hzr.params

	def get_init_state(self, src_context):
		return self.W_init(src_context[0,:,self.src_hidden_dim:])

	def __call__(self, new_index, prev_state):

		new_emb = self.embedder(new_index)
		new_ctx = self.context(prev_state)

		h_input, z_input, r_input = split(self.W_hzr(new_emb), 3) # (batch_size, 3*trg_hidden_dim) --> (batch_size,trg_hidden_dim)

		z_hidden, r_hidden = split(self.U_zr(prev_state), 2)

		h_context, z_context, r_context = split(self.C_hzr(new_ctx), 3)

		z = TT.nnet.sigmoid(z_input + z_hidden + z_context)
		r = TT.nnet.sigmoid(r_input + r_hidden + r_context)

		h_hidden = self.U_h(r*prev_state)

		proposed_h = TT.tanh(h_input + h_hidden + h_context)

		new_state = (1.-z) * prev_state + z * proposed_h # (batch_size, trg_hidden_dim)

		return new_state, new_emb, new_ctx


class QValue(Module):

	def __init__(self, vocab_size, trg_embedding_dim, src_hidden_dim, trg_hidden_dim, max_out_dim, n_max_out, name):
		super(QValue, self).__init__()
		self.name = name
		self.n_max_out = n_max_out

		self.U = Linear(trg_hidden_dim, max_out_dim, name=self.name+'_U')
		self.params += self.U.params

		self.V = Linear(trg_embedding_dim, max_out_dim, name=self.name+'_V', use_bias=False)
		self.params += self.V.params

		self.C = Linear(2*src_hidden_dim, max_out_dim, name=self.name+'_C', use_bias=False)
		self.params += self.C.params

		self.W = Linear(max_out_dim/n_max_out, vocab_size, name=self.name+'_W')
		self.params += self.W.params

	# def get_init_value(self, init_state):

	# 	proto_Q = self.U(init_state)
	# 	return TT.tanh(self.W( MaxOut(proto_Q, n_max=self.n_max_out) )) # (batch_size, vocab_size)

	def __call__(self, state, emb, ctx, pretrain=False):

		proto_Q = self.U(state) + self.V(emb) + self.C(ctx)
		Q_val = self.W( MaxOut(proto_Q, n_max=self.n_max_out) ) # (batch_size, vocab_size)
		if pretrain:
			return softmax(Q_val)
		return Q_val
