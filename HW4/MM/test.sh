#!/bin/sh

for i in `seq 1 20`; do
    diff=$(cat /nfs/data/data1_$i | ./run.sh | diff /nfs/data/ans1_$i -)
    if [ -z $diff ]; then
        echo data$i passed.
    else
        echo data$i failed.
        # echo $diff
    fi

done
