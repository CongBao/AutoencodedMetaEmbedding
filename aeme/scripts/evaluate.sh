#!/bin/bash

IN=${1:-"../results"}
OUT=${2:-"../evaluations"}

if [ ! -d $IN ]; then
    echo "Directory $IN does not exist"
    exit 0
fi

if [ ! -d $OUT ]; then
    mkdir $OUT
fi

for file in $IN/*.txt; do
    (cd repseval/src && exec python eval.py -m all -d 600 -i $file -o $OUT/${file%.txt}.csv)
done
