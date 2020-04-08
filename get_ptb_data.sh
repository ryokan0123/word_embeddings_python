#!/usr/bin/env bash

mkdir -p data/ptb
wget https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt -P data/ptb
wget https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt -P data/ptb