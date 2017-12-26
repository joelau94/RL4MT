#!/bin/sh

DEVICE=gpu7

if [ $1 = "-p" ]; then
	PYTHONPATH=../ THEANO_FLAGS=floatX=float32,device=$DEVICE,lib.cnmem=0.2 python run.py preprocess
fi

if [ $1 = "-t" ]; then
	if [ $2 = "s" ]; then
		PYTHONPATH=../ THEANO_FLAGS=floatX=float32,device=$DEVICE,lib.cnmem=0.2,on_unused_input=warn python run.py train supervised
	fi
	if [ $2 = "q" ]; then
		PYTHONPATH=../ THEANO_FLAGS=floatX=float32,device=$DEVICE,lib.cnmem=0.2,on_unused_input=warn python run.py train reinforcement
	fi
fi

if [ $1 = "-v" ]; then
	PYTHONPATH=../ THEANO_FLAGS=floatX=float32,device=$DEVICE,lib.cnmem=0.2,on_unused_input=warn python run.py validate
fi

if [ $1 = "-e" ]; then
	PYTHONPATH=../ THEANO_FLAGS=floatX=float32,device=$DEVICE,lib.cnmem=0.2,on_unused_input=warn python run.py evaluate
fi
