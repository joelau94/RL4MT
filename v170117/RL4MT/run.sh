#!/bin/sh

DEVICE=gpu3

if [ $1 = "-p" ]; then
	PYTHONPATH=../ THEANO_FLAGS=floatX=float32,device=$DEVICE,lib.cnmem=0.2 python run.py preprocess
fi

if [ $1 = "-t" ]; then
	if [ $2 = "a" ]; then
		PYTHONPATH=../ THEANO_FLAGS=floatX=float32,device=$DEVICE,lib.cnmem=0.2,on_unused_input=warn python run.py train actor
	fi
	if [ $2 = "c" ]; then
		PYTHONPATH=../ THEANO_FLAGS=floatX=float32,device=$DEVICE,lib.cnmem=0.2,on_unused_input=warn python run.py train critic
	fi
	if [ $2 = "ac" ]; then
		PYTHONPATH=../ THEANO_FLAGS=floatX=float32,device=$DEVICE,lib.cnmem=0.2,on_unused_input=warn,exception_verbosity=high python run.py train actor_critic
	fi
fi

if [ $1 = "-v" ]; then
	PYTHONPATH=../ THEANO_FLAGS=floatX=float32,device=$DEVICE,lib.cnmem=0.2,on_unused_input=warn python run.py validate
fi

if [ $1 = "-e" ]; then
	PYTHONPATH=../ THEANO_FLAGS=floatX=float32,device=$DEVICE,lib.cnmem=0.2,on_unused_input=warn python run.py evaluate
fi
