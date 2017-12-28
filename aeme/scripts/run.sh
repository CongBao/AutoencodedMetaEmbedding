#!/bin/bash

MODULE="/home/cong/fyp"
IN="/home/cong/data/CBOW-full.txt /home/cong/data/GloVe-full.txt"
OUT="../results"
RES="$OUT/${1:-"conc.txt"}"
MODEL=${2:-"conc ConcModel"}
TYPE=${3:-"MN"}
RATIO=${4:-"0.1"}

`export PATH=$PATH:/usr/local/cuda/bin`
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib:/usr/local/cuda/lib64`

if [ ! -d $OUT ]; then
    mkdir $OUT
fi

`python ../execute.py --module-path $MODULE -m $MODEL -i $IN -o $OUT -b 128 -e 1000 --noise-type $TYPE --noise-ratio $RATIO`
