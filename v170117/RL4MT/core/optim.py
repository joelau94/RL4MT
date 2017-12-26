import numpy as np
import theano
import theano.tensor as TT
from RL4MT.core.utils import *

class Optimizer(object):

	def __init__(self, name=None):
		self.name = name


class SGD(Optimizer):

	def __init__(self, inputs, costs, params, learning_rate, clipping, name=None):
		self.name = name
		self.params = params
		self.grads = [init_zeros(p.get_value().shape) for p in params]

		gradients = TT.grad(costs, params)
		grads_clip, grads_norm = clip(gradients, clipping, square=False)

		grads_upd = [(grads, new_grads) for grads, new_grads in zip(self.grads, grads_clip)]

		self.update_grads = theano.function(inputs, [costs, TT.sqrt(grads_norm)], updates=grads_upd)

		lr = np.float32(learning_rate)
		delta = [lr*grads for grads in self.grads]
		params_upd = [(p, p-d) for p, d in zip(self.params, delta)]

		self.update_params = theano.function([], [], updates=params_upd)


class AdaDelta(Optimizer):

	def __init__(self, inputs, costs, params, gamma, eps, clipping, name=None):
		super(AdaDelta, self).__init__()
		self.name = name
		self.params = params
		self.grads = [init_zeros(p.get_value().shape) for p in params]
		self.grads_sqr_avg = [init_zeros(p.get_value().shape) for p in params]
		self.delta_sqr_avg = [init_zeros(p.get_value().shape) for p in params]

		gradients = TT.grad(costs, params)
		grads_clip, grads_norm = clip(gradients, clipping)

		grads_upd = [(grads, new_grads) for grads, new_grads in zip(self.grads, grads_clip)]
		grads_sqr_avg_upd = [(grads_sqr_avg, gamma*grads_sqr_avg + (1.-gamma)*(new_grads**2.))
								for grads_sqr_avg, new_grads in zip(self.grads_sqr_avg, grads_clip) ]
		self.update_grads = theano.function(inputs, [costs, grads_norm], updates = grads_upd + grads_sqr_avg_upd)

		delta = [ grads * TT.sqrt(delta_sqr_avg+eps) / TT.sqrt(grads_sqr_avg+eps)
				for grads, delta_sqr_avg, grads_sqr_avg in zip(self.grads, self.delta_sqr_avg, self.grads_sqr_avg) ]

		delta_sqr_avg_upd = [ (delta_sqr_avg, gamma*delta_sqr_avg + (1.-gamma)*(new_delta**2.))
							for delta_sqr_avg, new_delta in zip(self.delta_sqr_avg, delta) ]
		params_upd = [(p,p-d) for p, d in zip(self.params, delta)]

		self.update_params = theano.function([], [], updates = params_upd + delta_sqr_avg_upd)


class ActorCriticAdaDelta(Optimizer):

	def __init__(self, inputs, actor_costs, critic_costs, actor_params, critic_params, gamma, eps, clipping, name=None):
		super(ActorCriticAdaDelta, self).__init__()
		self.name = name
		self.actor_params = actor_params
		self.critic_params = critic_params
		self.grads = [init_zeros(p.get_value().shape) for p in actor_params + critic_params]
		self.grads_sqr_avg = [init_zeros(p.get_value().shape) for p in actor_params + critic_params]
		self.delta_sqr_avg = [init_zeros(p.get_value().shape) for p in actor_params + critic_params]

		gradients = TT.grad(actor_costs, actor_params) + TT.grad(critic_costs, critic_params)
		grads_clip, grads_norm = clip(gradients, clipping)

		grads_upd = [(grads, new_grads) for grads, new_grads in zip(self.grads, grads_clip)]
		grads_sqr_avg_upd = [(grads_sqr_avg, gamma*grads_sqr_avg + (1.-gamma)*(new_grads**2.))
								for grads_sqr_avg, new_grads in zip(self.grads_sqr_avg, grads_clip) ]
		self.update_grads = theano.function(inputs, [actor_costs, critic_costs, grads_norm], updates = grads_upd + grads_sqr_avg_upd)

		delta = [ grads * TT.sqrt(delta_sqr_avg+eps) / TT.sqrt(grads_sqr_avg+eps)
				for grads, delta_sqr_avg, grads_sqr_avg in zip(self.grads, self.delta_sqr_avg, self.grads_sqr_avg) ]

		delta_sqr_avg_upd = [ (delta_sqr_avg, gamma*delta_sqr_avg + (1.-gamma)*(new_delta**2.))
							for delta_sqr_avg, new_delta in zip(self.delta_sqr_avg, delta) ]
		params_upd = [(p,p-d) for p, d in zip(actor_params+critic_params, delta)]

		self.update_params = theano.function([], [], updates = params_upd + delta_sqr_avg_upd)
