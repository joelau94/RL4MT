import os
import sys
import argparse
import re
import string
from glob import glob

from RL4MT.core.config import *
from RL4MT.core.data import *
from RL4MT.core.model import *

parser = argparse.ArgumentParser("Test best model")
parser.add_argument('-c', '--config')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-m', '--model_dir')
parser.add_argument('-s', '--source')
parser.add_argument('-r', '--reference_prefix')
parser.add_argument('-v', '--valid_dir')
parser.add_argument('-o', '--output_dir')

def get_bleu(eval_file):
	content = open(eval_file, 'r').read()
	return string.atof(re.findall('BLEU = (.*?),', content)[0])

if __name__ == "__main__":
	args = parser.parse_args()

	#init config
	glob_cfg = config_init()
	if args.config:
		glob_cfg = config_update(glob_cfg, json.load(open(args.config, 'r')))

	valid_files = glob(args.valid_dir+"*.eval")
	valid_files.sort()

	best_bleu = 0.
	best_mdl = ""

	for valid_file in valid_files:
		bleu = get_bleu(valid_file)
		model_file = model_dir + ''.join(valid_file.split('/')[-1].split('.')[:-1]) + '.npz'
		print("Model: {}, Validation BLEU: {}\n".format(model_file,bleu))
		if bleu >= best_bleu:
			best_bleu = bleu
			best_mdl = model_file

	if not best_mdl == "":
		trans_file = args.output_dir + ''.join(best_mdl.split('/')[-1].split('.')[:-1]) + '.trans'
		eval_file = args.output_dir + ''.join(best_mdl.split('/')[-1].split('.')[:-1]) + '.eval'
		cmd = 'THEANO_FLAGS=floatX=float32,device='+args.device
		cmd += ' python experiment/translate.py'
		cmd += ' -c experiment/config.json'
		cmd += ' -i ' + args.source
		cmd += ' -o ' + trans_file
		cmd += ' -m ' + best_mdl
		os.system(cmd)
		# score
		cmd = 'perl experiment/multi-bleu.perl -lc '
		cmd += args.reference_prefix
		cmd += ' < ' + trans_file
		cmd += ' > ' + eval_file
		os.system(cmd)
		test_bleu = get_bleu(eval_file)
		print("Best model: {}, Test BLEU: {}\n".format(best_mdl,test_bleu))
