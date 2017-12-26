import sys
import numpy as np
import theano
import theano.tensor as TT
import random

class SplitShapeError(Exception):
	def __init__(self, shape, splits):
		self.msg = "Unable to split shape ({}) into ({}) pieces !".format(shape,splits)

class MaxOutShapeError(Exception):
	def __init__(self, shape, n_max):
		self.msg = "Unable to divide shape ({}) for ({})-MaxOut !".format(shape,n_max)

class UndefinedTraineeError(Exception):
	def __init__(self, trainee):
		self.msg = "Undefined trainee: {}! Select a trainee ONLY from \"critic\" or \"actor_critic\"".format(trainee)

class UndefinedParameterError(Exception):
	def __init__(self, param_name):
		self.msg = "Undefined Parameter: {}! Load model failed.".format(param_name)

class IncompatibleParameterShapeError(Exception):
	def __init__(self, param_name, param_shape, expect_shape):
		self.msg = "Loaded parameter {} has shape {}, expects shape {} !".format(param_name,param_shape,expect_shape)

def init_weight(size, name, scale=0.01, shared=True):
	W = scale*np.random.randn(*size).astype('float32')
	if shared:
		return theano.shared(W,name=name)
	else:
		return W

def init_bias(size, name, scale=0.01, shared=True):
	b = scale*np.random.randn(size).astype('float32')
	if shared:
		return theano.shared(b,name=name)
	else:
		return b

def init_zeros(size, shared=True):
	t = np.zeros(size, dtype='float32')
	if shared:
		return theano.shared(t)
	else:
		return t

def clip(grads, threshold, square=True):
	grads_norm2 = sum(TT.sum(g**2) for g in grads)
	if square:
		grads_norm2 = TT.sqrt(grads_norm2)
	grads_clip = [TT.switch(TT.ge(grads_norm2,threshold),
	  				g/grads_norm2*threshold, g) 
					for g in grads]
	return grads_clip, grads_norm2

def split(_x, n):
	# only support 3d and 2d tensors
	input_size = _x.shape[-1]
	output_size = input_size/n
	output = []
	if _x.ndim == 3:
		for i in range(n):
			output.append(_x[:, :, i * output_size : (i+1) * output_size])
		return output
	else:
		for i in range(n):
			output.append(_x[:, i * output_size : (i+1) * output_size])
		return output
	# try:
	# 	if input_size % n != 0:
	# 		raise SplitShapeError(input_size, n)
	# 	output_size = input_size/n
	# 	output = []
	# 	if _x.ndim == 3:
	# 		for i in range(n):
	# 			output.append(_x[:, :, i * output_size : (i+1) * output_size])
	# 		return output
	# 	else:
	# 		for i in range(n):
	# 			output.append(_x[:, i * output_size : (i+1) * output_size])
	# 		return output
	# except SplitShapeError, e:
	# 	print e.msg
	# 	sys.exit(1)

def MaxOut(_input, n_max=2):
	batch_size = _input.shape[0]
	vector_size = _input.shape[1]
	return _input.reshape( (batch_size, vector_size/n_max, n_max) ).max(2)
	# try:
	# 	if vector_size % n_max != 0:
	# 		raise MaxOutShapeError(vector_size, n_max)
	# 	return _input.reshape( (batch_size, vector_size/n_max, n_max) ).max(2)
	# except MaxOutShapeError, e:
	# 	print e.msg
	# 	sys.exit(1)

def softmax(energy, axis=1):
	exp = TT.exp( energy - TT.max(energy, axis=1).reshape((energy.shape[0],1)) )
	normalizer = TT.sum(exp, axis=axis).reshape((energy.shape[0],1))
	return exp/normalizer

def softmax3d(energy):
	energy_reshape = energy.reshape( (energy.shape[0]*energy.shape[1], energy.shape[2]) )
	return TT.nnet.softmax(energy_reshape).reshape( (energy.shape[0], energy.shape[1], energy.shape[2]) )

def one_hot(index, n_dim):
	""" Accepts a list of indices """
	one_hots = np.zeros( (len(index), n_dim), dtype='float32' )
	for i in range(len(index)):
		one_hots[i,index[i]] = 1.
	return one_hots