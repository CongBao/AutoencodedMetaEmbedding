#!/bin/bash

IN="~/new_results"
OUT="~/new_eval_results"

for file in $IN/*.txt; do
    (cd ~/aeme/aeme/scripts/repseval/src && exec python eval.py -m all -d 600 -i "$file" -o "$OUT/$(file%.txt).csv")
done
