import sys
import os.path
import numpy as np
import random
import argparse
import cPickle as pkl
import re

from RL4MT.core.model import *
from RL4MT.core.data import *
from RL4MT.experiment.preprocess import *

parser = argparse.ArgumentParser("File Translation.")
parser.add_argument('-c', '--config')
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o','--output', required=True)
parser.add_argument('-m', '--model', required=True)
parser.add_argument('-b', '--beam_size', type=int, default=5)
parser.add_argument('-e', '--explore_ratio', type=int, default=5)
parser.add_argument('-s', '--step_max', type=int, default=200)

if __name__ == "__main__":
	args = parser.parse_args()

	#init config
	glob_cfg = config_init()
	if args.config:
		glob_cfg = config_update(glob_cfg, json.loads(open(args.config, 'r').read()))

	# load dict and model
	src_w2i = load_dict(glob_cfg['src_w2i'])
	trg_i2w = load_dict(glob_cfg['trg_i2w'])
	mdl = ActorCriticModel(glob_cfg['src_vocab_size'], glob_cfg['trg_vocab_size'],
							glob_cfg['src_embedding_dim'], glob_cfg['trg_embedding_dim'],
							glob_cfg['src_hidden_dim'], glob_cfg['trg_hidden_dim'],
							glob_cfg['attn_dim'], glob_cfg['src_max_len'], glob_cfg['trg_max_len'], glob_cfg['buffer_size'],
							glob_cfg['n_max_out'], glob_cfg['trg_nullsym'], glob_cfg['discount'], "RL4MT")
	mdl.load(args.model)
	mdl.TestPhase(beam_size=args.beam_size,  explore_ratio=args.explore_ratio)

	# input and output files
	trans_in = open(args.input,'r')
	trans_out = open(args.output,'w')

	# translate
	for line in trans_in:
		src_sent = line.strip().split()
		src_sent_index = sentence_w2i(src_sent, src_w2i, glob_cfg['src_unksym'])
		trg_sent_index = mdl.translate(src_sent_index, args.step_max, args.beam_size, args.explore_ratio)
		trg_sent = sentence_i2w(trg_sent_index, trg_i2w, glob_cfg['trg_nullsym'])
		trans_out.write(trg_sent+'\n')