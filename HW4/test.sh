#!/bin/bash


num=$1
name=$2
val=0.0

echo "Run 10 times with $num threads...."
for i in `seq 1 10`; do
    param=$(mpirun -np $num --hostfile hosts6 $name 1000000000 | awk '{ if (NR==1) print $1; else print $4}')
    read -r ans sec <<<$(echo $param)
    echo "Result: $ans"
    val=$(echo -e $val $sec | awk '{ print ($1+$2) }')
done
val=$(echo $val $num | awk '{ print $1/$2 }' )
echo "Average: $val secs"