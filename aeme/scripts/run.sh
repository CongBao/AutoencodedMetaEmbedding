#!/bin/bash

IN="~/data/CBOW-full.txt ~/data/GloVe-full.txt"
OUT="~/new_results/ae_SP_5.txt"
MODEL="ae AEModel"

`rm nohup.out`
`nohup python ../execute.py --module-path ~/aeme -m $MODEL -i $IN -o $OUT -b 128 -e 1000 --noise-type SP --noise-ratio 0.05 &`