#!/bin/bash

IN=${1:-"../results"}

if [ ! -d $IN ]; then
    echo "Directory $IN does not exist"
    exit 0
fi

for file in $IN/*.txt; do
    echo "File to process: $file"
    if [ -f ${file%.txt}.csv ]; then
        echo "File $file has been evaluated, skip it"
        continue
    fi
    (cd repseval/src && exec python eval.py -m all -d 200 -i $file -o ${file%.txt}.csv)
done
