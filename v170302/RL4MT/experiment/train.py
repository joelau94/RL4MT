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

log_freq = 100

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
	mdl = DeepQModel(glob_cfg['src_vocab_size'], glob_cfg['trg_vocab_size'],
							glob_cfg['src_embedding_dim'], glob_cfg['trg_embedding_dim'],
							glob_cfg['src_hidden_dim'], glob_cfg['trg_hidden_dim'],
							glob_cfg['attn_dim'], glob_cfg['max_out_dim'],
							glob_cfg['n_max_out'], glob_cfg['discount'], "RL4MT")
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
	if args.phase == 'supervised':

		mdl.SupervisedPhase(glob_cfg['batch_size'])
		trainer = AdaDelta(mdl.inputs, mdl.costs, mdl.params, glob_cfg['gamma'], glob_cfg['eps'], glob_cfg['clipping'], name='Supervised_Pretrain')

		data.set_mode("batch")

		log.write("Pre-training begins ({})\n".format( str(datetime.datetime.now()) ) )
		log.flush()

		while True:
			x, x_mask, y, y_mask = data.next()
			while x.shape[0] == 0:
				x, x_mask, y, y_mask = data.next()
			
			if data.get_iternum() % log_freq == 0:
				log.write("Pre-training: Iteration {} begins ({})\n".format( data.get_iternum(), str(datetime.datetime.now()) ) )
				log.flush()

			# save checkpoint
			if data.get_iternum() % glob_cfg['save_checkpoint_freq'] == 0:
				mdl.save(glob_cfg['checkpoint_model'])
				data.save_status(glob_cfg['checkpoint_status'])

			# save model
			if data.get_iternum() % glob_cfg['save_model_freq'] == 0:
				model_name = "%s%04.i%s" % (glob_cfg['save_path']+'model.', data.get_iternum()/glob_cfg['save_model_freq'], '.npz')
				status_name = "%s%04.i%s" % (glob_cfg['save_path']+'status.', data.get_iternum()/glob_cfg['save_model_freq'], '.pkl')
				mdl.save(model_name)
				data.save_status(status_name)
				train_eval(model_name, eval_log)

			# debug_fn = theano.function(mdl.inputs,[mdl.y_reorder])
			# debug_res = debug_fn(x,x_mask,y)
			# print("y:{}\ny_reorder:{}\n".format(y,debug_res[0]))

			costs, grads_norm = trainer.update_grads(x, x_mask, y, y_mask)
			if data.get_iternum() % log_freq == 0:
				log.write("Costs:{}\n".format(costs))
				log.flush()
			# nan/inf process
			if np.isinf(costs.mean()) or np.isnan(costs.mean()):
				log.write('Nan/Inf while training (Costs:{}), saving model and exit. ({})\n'.format( costs, str(datetime.datetime.now()) ))
				log.flush()
				mdl.save(glob_cfg['checkpoint_model'])
				data.save_status(glob_cfg['checkpoint_status'])
				sys.exit(1)
			trainer.update_params()

	# reinforcement phase
	elif args.phase == 'reinforcement':

		mdl.ReinforcementPhase(glob_cfg['batch_size'], glob_cfg['trg_max_len'])
		trainer = AdaDelta(mdl.inputs, mdl.costs, mdl.params, glob_cfg['gamma'], glob_cfg['eps'], glob_cfg['clipping'], name='Q_learning')

		data.set_mode("batch")

		log.write("Q-learning begins ({})\n".format( str(datetime.datetime.now()) ) )
		log.flush()

		while True:
			x, x_mask, y, y_mask = data.next()
			while x.shape[0] == 0:
				x, x_mask, y, y_mask = data.next()

			greedy_eps = max(glob_cfg['greedy_eps_min'], glob_cfg['greedy_eps']-data.get_iternum()*glob_cfg['greedy_anneal'])
			
			if data.get_iternum() % log_freq == 0:
				log.write("Q-learning: Iteration {} begins, greedy_eps={} ({})\n".format( data.get_iternum(), greedy_eps, str(datetime.datetime.now()) ) )
				log.flush()

			# save checkpoint
			if data.get_iternum() % glob_cfg['save_checkpoint_freq'] == 0:
				mdl.save(glob_cfg['checkpoint_model'])
				data.save_status(glob_cfg['checkpoint_status'])

			# save model
			if data.get_iternum() % glob_cfg['save_model_freq'] == 0:
				model_name = "%s%04.i%s" % (glob_cfg['save_path']+'model.', data.get_iternum()/glob_cfg['save_model_freq'], '.npz')
				status_name = "%s%04.i%s" % (glob_cfg['save_path']+'status.', data.get_iternum()/glob_cfg['save_model_freq'], '.pkl')
				mdl.save(model_name)
				data.save_status(status_name)
				train_eval(model_name, eval_log)

			# debug_fn = theano.function(mdl.inputs,[mdl.y_reorder])
			# debug_res = debug_fn(x,x_mask,y)
			# print("y:{}\ny_reorder:{}\n".format(y,debug_res[0]))

			y_sampl = mdl.sample(x, x_mask, greedy_eps)[0]
			y_mask_sampl = mdl.get_y_mask(y_sampl)
			rewards = mdl.get_reward(y, y_mask, y_sampl)

			costs, grads_norm = trainer.update_grads(x, x_mask, y_sampl, y_mask_sampl, rewards)
			if data.get_iternum() % log_freq == 0:
				log.write("Costs:{}\n".format(costs))
				log.flush()
			# nan/inf process
			if np.isinf(costs.mean()) or np.isnan(costs.mean()):
				log.write('Nan/Inf while training (Costs:{}), saving model and exit. ({})\n'.format( costs, str(datetime.datetime.now()) ))
				log.flush()
				mdl.save(glob_cfg['checkpoint_model'])
				data.save_status(glob_cfg['checkpoint_status'])
				sys.exit(1)
			trainer.update_params()
