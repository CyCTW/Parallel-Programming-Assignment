#!/bin/sh
for i in $(seq 1 10); do
	./mandelbrot -i 1000 -g 1
done
