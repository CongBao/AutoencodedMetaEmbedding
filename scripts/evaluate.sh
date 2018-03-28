#!/bin/bash

IN=$1

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
    (cd repseval/src && exec python eval.py -m all -d $2 -i $file -o ${file%.txt}.csv)
done
