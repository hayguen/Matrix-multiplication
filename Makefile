
.PHONY: test test-mp test-naive test-naive-mp clean

ALL:	matrix-block-mp matrix-naive-mp matrix-block matrix-naive

matrix-block-mp:	matrix-tiling.c
	gcc -O3 -march=native -fopenmp matrix-tiling.c -o matrix-block-mp

matrix-block:	matrix-tiling.c
	gcc -O3 -march=native matrix-tiling.c -o matrix-block

matrix-naive-mp:	matrix.c
	gcc -O3 -march=native -fopenmp matrix.c -o matrix-naive-mp

matrix-naive:	matrix.c
	gcc -O3 -march=native matrix.c -o matrix-naive

clean:
	rm matrix-block-mp matrix-naive-mp matrix-block matrix-naive

test-mp:	matrix-block-mp
	perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses ./matrix-block-mp

test:	matrix-block
	perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses ./matrix-block

test-naive:	matrix-naive
	perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses ./matrix-naive

test-naive-mp:	matrix-naive-mp
	perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses ./matrix-naive-mp
