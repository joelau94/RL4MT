import os
import sys
import argparse
import string
import re

from RL4MT.core.config import *
from RL4MT.core.data import *
from RL4MT.core.model import *

parser = argparse.ArgumentParser("Test a given model.")
parser.add_argument('-c', '--config')
parser.add_argument('-d', '--device', default='gpu5')
parser.add_argument('-m', '--model')
parser.add_argument('-i', '--input')
parser.add_argument('-r', '--reference_prefix')
parser.add_argument('-o', '--output')
parser.add_argument('-e', '--eval_file')

def get_bleu(eval_file):
	content = open(eval_file, 'r').read()
	return string.atof(re.findall('BLEU = (.*?),', content)[0])

if __name__ == "__main__":
	args = parser.parse_args()

	#init config
	glob_cfg = config_init()
	if args.config:
		glob_cfg = config_update(glob_cfg, json.load(open(args.config, 'r')))

	cmd = 'THEANO_FLAGS=floatX=float32,device='+args.device
	cmd += ' python experiment/translate.py'
	cmd += ' -c experiment/config.json'
	cmd += ' -i ' + args.input
	cmd += ' -o ' + args.output
	cmd += ' -m ' + args.model
	os.system(cmd)
	cmd = 'perl experiment/multi-bleu.perl -lc '
	cmd += args.reference_prefix
	cmd += ' < ' + args.output
	cmd += ' > ' + args.eval_file
	os.system(cmd)
	print("Model: {}, BLEU: {}".format(args.model,get_bleu(args.eval_file)))