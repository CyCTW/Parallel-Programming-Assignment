#! /bin/bash

avg=0.0
for i in {1..20}; do
    result=$(mpirun -np 4 --hostfile /nfs/.grade/HW4/mat1_hosts matmul < /nfs/data/data1_$i | diff /nfs/data/ans1_$i - | tail -n +2)
    if [[ $(echo "$result" | wc -l) -ne 4 ]]; then
        echo "[running data1_$i]: Wrong Answer"
        continue
    fi
    runtime=$(echo "$result" | awk '{ print $5 }' | sort -n | tail -n 1)
    echo "[running data1_$i]: Accepted, $runtime"
    avg=$(awk '{print $1 + $2}' <<< "${avg} ${runtime}")
done
echo $(awk '{print $1/20}' <<< "${avg}")

avg=0.0
for i in {1..20}; do
    result=$(mpirun -np 5 --hostfile /nfs/.grade/HW4/mat2_hosts matmul < /nfs/data/data2_$i | diff /nfs/data/ans2_$i - | tail -n +2)
    if [[ $(echo "$result" | wc -l) -ne 5 ]]; then
        echo "[running data2_$i]: Wrong Answer"
        continue
    fi
    runtime=$(echo "$result" | awk '{ print $5 }' | sort -n | tail -n 1)
    echo "[running data2_$i]: Accepted, $runtime"
    avg=$(awk '{print $1 + $2}' <<< "${avg} ${runtime}")
done
echo $(awk '{print $1/20}' <<< "${avg}")
