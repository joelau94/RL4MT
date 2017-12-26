import os
import sys
import argparse
from glob import glob
import re

from RL4MT.core.config import *

parser = argparse.ArgumentParser("Batch Validation")
parser.add_argument('-c', '--config')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-m', '--model_dir')
parser.add_argument('-s', '--source')
parser.add_argument('-r', '--reference_prefix')
parser.add_argument('-o', '--output_dir')

if __name__ == "__main__":
	args = parser.parse_args()

	#init config
	glob_cfg = config_init()
	if args.config:
		glob_cfg = config_update(glob_cfg, json.load(open(args.config, 'r')))

	mdl_files = glob(args.model_dir+"*.npz")
	mdl_files.sort()
	for mdl_file in mdl_files:
		# translate
		trans_file = args.output_dir + ''.join(mdl_file.split('/')[-1].split('.')[:-1]) + '.trans'
		eval_file = args.output_dir + ''.join(mdl_file.split('/')[-1].split('.')[:-1]) + '.eval'
		cmd = 'THEANO_FLAGS=floatX=float32,device='+args.device
		cmd += ' python experiment/translate.py'
		cmd += ' -c experiment/config.json'
		cmd += ' -i ' + args.source
		cmd += ' -o ' + trans_file
		cmd += ' -m ' + mdl_file
		os.system(cmd)
		# score
		cmd = 'perl experiment/multi-bleu.perl -lc '
		cmd += args.reference_prefix
		cmd += ' < ' + trans_file
		cmd += ' > ' + eval_file
		os.system(cmd)
