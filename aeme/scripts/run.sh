#!/bin/bash

IN="~/data/CBOW-full.txt ~/data/GloVe-full.txt"
OUT="~/new_results/ae_SP_5.txt"
MODEL="ae AEModel"

`export PATH=$PATH:/usr/local/cuda/bin`
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib:/usr/local/cuda/lib64`

if [ -f "nohup.out" ]; then
    `rm nohup.out`
fi

`nohup python ../execute.py --module-path ~/aeme -m $MODEL -i $IN -o $OUT -b 128 -e 1000 --noise-type SP --noise-ratio 0.05 &`
