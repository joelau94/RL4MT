import sys
import numpy as np
import theano
import theano.tensor as TT
import cPickle as pkl
import json
import argparse
import signal
import datetime
import subprocess

from RL4MT.core.config import *
from RL4MT.core.data import *
from RL4MT.core.model import *
from RL4MT.core.optim import *
from RL4MT.experiment.test import *

parser = argparse.ArgumentParser("Training of RL4MT.")
parser.add_argument('-c', '--config')
parser.add_argument('-p', '--phase')
parser.add_argument('-r', '--resume_model')
parser.add_argument('-d', '--data_status')

def train_eval(model, log):
	subprocess.Popen(["python", "experiment/test_any_trans.py", "-c", "experiment/config.json", "-m", model,
						"-i", "data/valid_set/nist02/nist02.cn", "-r", "data/valid_set/nist02/nist02.en",
						"-o", 'train_eval/'+''.join(model.split('/')[-1].split('.')[:-1])+'.trans',
						"-e", 'train_eval/'+''.join(model.split('/')[-1].split('.')[:-1])+'.eval'], stdout=log)

if __name__ == '__main__':
	args = parser.parse_args()

	if args.phase is None:
		print("Please specify training phase.")
		sys.exit(1)

	glob_cfg = config_init()
	if args.config:
		glob_cfg = config_update(glob_cfg, json.load(open(args.config, 'r')))

	log = open(glob_cfg['train_log'],'a+')
	eval_log = open(glob_cfg['train_eval'],'a+')

	# load data
	log.write('Load training data ({})\n'.format( str(datetime.datetime.now()) ))
	log.flush()
	data_config = { 'src_shuf': glob_cfg['src_shuf'],
					'trg_shuf': glob_cfg['trg_shuf'],
					'src_max_len': glob_cfg['src_max_len'],
					'trg_max_len': glob_cfg['trg_max_len'],
					'peek_num': glob_cfg['peek_num'],
					'batch_size': glob_cfg['batch_size'] }
	data = Dataset(data_config)
	if args.data_status:
		data.load_status(args.data_status)

	# initialize model
	log.write('Initializing model ({})\n'.format( str(datetime.datetime.now()) ))
	log.flush()
	mdl = ActorCriticModel(glob_cfg['src_vocab_size'], glob_cfg['trg_vocab_size'],
							glob_cfg['src_embedding_dim'], glob_cfg['trg_embedding_dim'],
							glob_cfg['src_hidden_dim'], glob_cfg['trg_hidden_dim'],
							glob_cfg['attn_dim'], glob_cfg['src_max_len'], glob_cfg['trg_max_len'], glob_cfg['buffer_size'],
							glob_cfg['n_max_out'], glob_cfg['trg_nullsym'], glob_cfg['discount'], "RL4MT")
	if args.resume_model:
		mdl.load(args.resume_model)

	# grace exit
	def grace_exit(signum, frame):
		mdl.save(glob_cfg['checkpoint_model'])
		data.save_status(glob_cfg['checkpoint_status'])
		log.write("Signal {} received. Checkpoint saved. ({})\n".format( signum, str(datetime.datetime.now()) ))
		sys.exit(0)

	# capture signals
	signal.signal(signal.SIGINT, grace_exit)
	signal.signal(signal.SIGTERM, grace_exit)
	signal.signal(signal.SIGABRT, grace_exit)
	signal.signal(signal.SIGFPE, grace_exit)
	signal.signal(signal.SIGILL, grace_exit)
	signal.signal(signal.SIGSEGV, grace_exit)

	# supervised phase
	if args.phase == 'actor':

		mdl.ActorPretrainPhase(glob_cfg['batch_size'])
		actor_params = mdl.encoder.params + mdl.actor.action_word_params

		actor_pretrainer = AdaDelta(mdl.inputs, mdl.costs, actor_params, glob_cfg['gamma'], glob_cfg['eps'], glob_cfg['clipping'], name='Actor_Pretrain')

		data.set_mode("batch")

		while True:
			x, x_mask, y = data.next()
			while x.shape[0] == 0:
				x, x_mask, y = data.next()

			y_reorder, buffer_indices, positions = mdl.actor_pretrain_sequence(y)
			
			log.write("Pre-training (Actor): Iteration {} begins ({})\n".format( data.get_iternum(), str(datetime.datetime.now()) ) )
			log.flush()

			# save checkpoint
			if data.get_iternum() % glob_cfg['save_checkpoint_freq'] == 0:
				mdl.save(glob_cfg['checkpoint_model'])
				data.save_status(glob_cfg['checkpoint_status'])

			# save model
			if data.get_iternum() % glob_cfg['save_model_freq'] == 0:
				model_name = "%s%04.i%s" % (glob_cfg['save_path']+'model.', data.get_iternum()/glob_cfg['save_model_freq'], '.npz')
				mdl.save(model_name)
				train_eval(model_name, eval_log)

			# debug_fn = theano.function(mdl.inputs,[mdl.y_reorder])
			# debug_res = debug_fn(x,x_mask,y)
			# print("y:{}\ny_reorder:{}\n".format(y,debug_res[0]))

			costs, grads_norm = actor_pretrainer.update_grads(x, x_mask, y_reorder, buffer_indices, positions)
			log.write("Costs:{}\n".format(costs))
			log.flush()
			# nan/inf process
			if np.isinf(costs.mean()) or np.isnan(costs.mean()):
				log.write('Nan/Inf while training, saving model and exit. ({})\n'.format( str(datetime.datetime.now()) ))
				log.flush()
				mdl.save(glob_cfg['checkpoint_model'])
				sys.exit(1)
			actor_pretrainer.update_params()

	elif args.phase == 'critic':

		mdl.CriticPretrainPhase(glob_cfg['batch_size'])
		critic_pretrainer = AdaDelta(mdl.inputs, mdl.costs, mdl.critic.params, glob_cfg['gamma'], glob_cfg['eps'], glob_cfg['clipping'], name='Critic_Pretrain')

		data.set_mode("batch")

		while True:
			x, x_mask, y = data.next()
			while x.shape[0] == 0:
				x, x_mask, y = data.next()

			buffer_indices, rewards = mdl.critic_pretrain_sequence(y)

			log.write("Pre-training (Critic): Iteration {} begins ({})\n".format( data.get_iternum(), str(datetime.datetime.now()) ) )
			log.flush()

			# save checkpoint
			if data.get_iternum() % glob_cfg['save_checkpoint_freq'] == 0:
				mdl.save(glob_cfg['checkpoint_model'])
				data.save_status(glob_cfg['checkpoint_status'])

			# save model
			if data.get_iternum() % glob_cfg['save_model_freq'] == 0:
				model_name = "%s%04.i%s" % (glob_cfg['save_path']+'model.', data.get_iternum()/glob_cfg['save_model_freq'], '.npz')
				mdl.save(model_name)
				train_eval(model_name, eval_log)
			
			costs, grads_norm = critic_pretrainer.update_grads(x, x_mask, buffer_indices, rewards)
			log.write("Costs:{}\n".format(costs))
			log.flush()
			# nan/inf process
			if np.isinf(costs.mean()) or np.isnan(costs.mean()):
				log.write('Nan/Inf while training, saving model and exit. ({})\n'.format( str(datetime.datetime.now()) ))
				log.flush()
				mdl.save(glob_cfg['checkpoint_model'])
				sys.exit(1)
			critic_pretrainer.update_params()

	elif args.phase == "actor_critic":

		mdl.ActorCriticTrainingPhase(glob_cfg['batch_size'])
		actor_params = mdl.actor.params + mdl.encoder.params
		critic_params = mdl.critic.params
		ac_trainer = ActorCriticAdaDelta(mdl.dec_inputs, mdl.actor_costs, mdl.critic_costs, actor_params, critic_params, glob_cfg['gamma'], glob_cfg['eps'], glob_cfg['clipping'], name='Actor_Critic_Policy_Gradient')

		data.set_mode("batch")

		while True:
			x, x_mask, y = data.next()
			while x.shape[0] == 0:
				x, x_mask, y = data.next()

			# debug_res = mdl.debug_fn(x,x_mask)
			# print("new_state:{}\n".format(debug_res[0].shape))

			log.write("Training (actor-critic): Iteration {} begins ({})\n".format( data.get_iternum(), str(datetime.datetime.now()) ) )
			log.flush()

			# save checkpoint
			if data.get_iternum() % glob_cfg['save_checkpoint_freq'] == 0:
				mdl.save(glob_cfg['checkpoint_model'])
				data.save_status(glob_cfg['checkpoint_status'])

			# save model
			if data.get_iternum() % glob_cfg['save_model_freq'] == 0:
				model_name = "%s%04.i%s" % (glob_cfg['save_path']+'model.', data.get_iternum()/glob_cfg['save_model_freq'], '.npz')
				mdl.save(model_name)
				train_eval(model_name, eval_log)

			# sample
			decode_buffers, positions, words = mdl.sample(x, x_mask, glob_cfg['max_step_training'])

			# compute rewards
			rewards = mdl.get_rewards(y, decode_buffers)

			# update gradients
			actor_costs, critic_costs, grads_norm = ac_trainer.update_grads(x, x_mask, decode_buffers, rewards, positions, words)
			log.write("Actor costs: {}; Critic costs: {}.\n".format(actor_costs, critic_costs))
			log.flush()
			# nan/inf process
			mean_costs = actor_costs.mean() + critic_costs.mean()
			if np.isinf(mean_costs) or np.isnan(mean_costs):
				log.write('Nan/Inf while training, saving model and exit. ({})\n'.format( str(datetime.datetime.now()) ))
				log.flush()
				mdl.save(glob_cfg['checkpoint_model'])
				sys.exit(1)
			ac_trainer.update_params()
