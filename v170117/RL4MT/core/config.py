import numpy
import json

def config_init():
	config = {}

	# data file paths
	config['src_text'] = 'data/train.zh'
	config['trg_text'] = 'data/train.en'
	config['src_index'] = 'data/train.index.zh'
	config['trg_index'] = 'data/train.index.en'
	config['src_shuf'] = 'data/train.shuf.zh'
	config['trg_shuf'] = 'data/train.shuf.en'
	config['src_i2w'] = 'data/i2w.zh'
	config['trg_i2w'] = 'data/i2w.en'
	config['src_w2i'] = 'data/w2i.zh'
	config['trg_w2i'] = 'data/w2i.en'

	# vocab
	config['src_nullsym'] = 0
	config['trg_nullsym'] = 0
	config['src_unksym'] = 1
	config['trg_unksym'] = 1
	config['src_vocab_size'] = 30000
	config['trg_vocab_size'] = 30000

	# model hyper-parameters
	config['src_embedding_dim'] = 160
	config['trg_embedding_dim'] = 160
	config['src_hidden_dim'] = 280
	config['trg_hidden_dim'] = 280
	config['attn_dim'] = 280
	config['buffer_size'] = 100
	config['discount'] = 0.9
	config['n_max_out'] = 2

	# data
	config['src_max_len'] = 50
	config['trg_max_len'] = 50
	config['peek_num'] = 20
	config['batch_size'] = 60
	config['buffer_size'] = 100
	config['max_step_training'] = 200

	# checkpoint & save model
	config['save_model_freq'] = 10000 # iters (updates)
	config['save_checkpoint_freq'] = 2000 # iters (updates)
	config['save_path'] = 'models/'
	config['checkpoint_model'] = 'models/checkpoint_model.npz'
	config['checkpoint_status'] = 'experiment/checkpoint_status.pkl'

	# optimizing
	config['lr'] = 1.
	config['gamma'] = 0.95
	config['eps'] = 1e-6
	config['clipping'] = 1.

	# log file
	config['train_log'] = "train_log.txt"	
	config['train_eval'] = "train_eval.txt"

	return config

def config_update(config, dic):
	for key in dic:
		config[key] = dic[key]
	return config

def save_config(config, path):
	with open(path, 'w+') as f:
		json.dump(config, f)