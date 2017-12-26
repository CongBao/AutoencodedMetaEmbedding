#!/bin/bash
#$1 filename $2 fileid $3 basepath
filename=$1
fileid=$2
basepath=$3
perl maxdiff_to_scale.pl $basepath/Phase2Answers/Phase2Answers-$fileid.txt ../work/semeval-tmp/TurkerScaled-$fileid.txt
perl score_scale.pl ../work/semeval-tmp/TurkerScaled-$fileid.txt $filename ../work/semeval-tmp/SpearmanRandomScaled-$fileid.txt
perl scale_to_maxdiff.pl $basepath/Phase2Questions/Phase2Questions-$fileid.txt $filename ../work/semeval-tmp/MaxDiff-$fileid.txt
perl score_maxdiff.pl $basepath/Phase2Answers/Phase2Answers-$fileid.txt ../work/semeval-tmp/MaxDiff-$fileid.txt ../work/semeval-tmp/MaxDiffFinal-$fileid.txt