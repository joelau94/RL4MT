import sys
import numpy as np
import cPickle as pkl
import random
from collections import Counter
import argparse
from RL4MT.core.config import *

parser = argparse.ArgumentParser("Data preprocessing.")
parser.add_argument('-c', '--config')

def build_dict(vocab_size, txt_file, i2w_file, w2i_file):

	word_freq = Counter([ word for word in open(txt_file,'r').read().strip().split() ])

	i2w_list = ['<null>', '<unk>'] + list( dict(word_freq.most_common(vocab_size-2)).keys() )
	w2i_dict = dict( [ (i2w_list[i], i) for i in xrange(len(i2w_list)) ] )

	pkl.dump(i2w_list, open(i2w_file,'wb'))
	pkl.dump(w2i_dict, open(w2i_file,'wb'))

	return len(word_freq)

def corpus2index(txt_file, w2i_file, corpus_file):

	w2i_dict = pkl.load(open(w2i_file,'rb'))
	corpus = [ [ w2i_dict[word] if w2i_dict.has_key(word) else w2i_dict['<unk>'] for word in line.strip().split() ]
				for line in open(txt_file,'r').readlines() ]
	pkl.dump(corpus, open(corpus_file,'wb'))

	return len(corpus)

def shuffle(src_corpus_file, trg_corpus_file, src_shuf_file, trg_shuf_file):

	src = pkl.load(open(src_corpus_file,'rb'))
	trg = pkl.load(open(trg_corpus_file,'rb'))
	assert len(src) == len(trg)

	order = range(len(src))
	random.shuffle(order)
	src_shuf = [ src[i] for i in order ]
	trg_shuf = [ trg[i] for i in order ]

	pkl.dump(src_shuf, open(src_shuf_file,'wb'))
	pkl.dump(trg_shuf, open(trg_shuf_file,'wb'))

# for other scripts to call (dict files use relative paths)
def load_dict(dict_file):
	return pkl.load(open(dict_file,'rb'))


# main
if __name__ == "__main__":
	args = parser.parse_args()

	#init config
	glob_cfg = config_init()
	if args.config:
		glob_cfg = config_update(glob_cfg, json.load(open(args.config, 'r')))

	print("Begin Preprocessing ...\n")
	print("Source: {} unique words, {} in vocab.\n".format( build_dict(glob_cfg['src_vocab_size'], glob_cfg['src_text'], glob_cfg['src_i2w'], glob_cfg['src_w2i']), glob_cfg['src_vocab_size']) )
	print("Target: {} unique words, {} in vocab.\n".format( build_dict(glob_cfg['trg_vocab_size'], glob_cfg['trg_text'], glob_cfg['trg_i2w'], glob_cfg['trg_w2i']), glob_cfg['trg_vocab_size']) )
	print("Source: {} sentences.\n".format( corpus2index(glob_cfg['src_text'], glob_cfg['src_w2i'], glob_cfg['src_index']) ))
	print("Target: {} sentences.\n".format( corpus2index(glob_cfg['trg_text'], glob_cfg['trg_w2i'], glob_cfg['trg_index']) ))
	shuffle(glob_cfg['src_index'], glob_cfg['trg_index'], glob_cfg['src_shuf'], glob_cfg['trg_shuf'])
	print("Shuffle done.\n")