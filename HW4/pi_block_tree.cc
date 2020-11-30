#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include<random>

typedef long long lld;

unsigned R = (1<<15)-1; // 0x7FFF

int xorshift(int &state) {
	int x = state;
	x ^= (x << 13);
	x ^= (x >> 17);
	x ^= (x << 5);
	return state = x;
}

lld count_toss(lld iterations) {
	std::random_device rd;
	lld toss;
	lld cnt = 0;

	int rdn = rd();
	for(toss = 0; toss < iterations; toss++) {
		unsigned xy = xorshift( rdn );
		unsigned x = (xy & 0x7FFF0000) >> 16;
		unsigned y = xy & 0x00007FFF;
		
		cnt += ( (x*x + y*y) <= R*R);
	}
	

	return cnt;
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // TODO: binary tree redunction
	int bitmsk = 0;
	int times = log2(world_size);
	/*
	   0: 0000
	   1: 0001
	   2: 0010
	   3: 0011
	   4: 0100
	   5: 0101
	   6: 0110
	   7: 0111
			  */
	int tag = 0;
	lld count = 0;
	lld iterations = tosses / world_size;
	int send = 0;
	count = count_toss( iterations );
	while(bitmsk < times ) {
		int mask = ( 1 << bitmsk );
		printf("rank: %d ", world_rank);

		// 1: send, 0: recv
		if (world_rank & mask) {
			// count += count_toss( iterations );
			int dest = world_rank & (~mask);
			MPI_Send(&count, 1, MPI_LONG_LONG, dest, tag, MPI_COMM_WORLD);
			send++;
			//printf("send to %d\n", dest);
			break;
		} else {
			// count += count_toss( iterations );
			lld src_count;
			int dest = world_rank | mask;
			MPI_Recv(&src_count, 1, MPI_LONG_LONG, dest, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			count += src_count;
			//printf("recv from %d\n", dest);
		}
		bitmsk++;
	}
	printf("Send times: %d\n", send);	

    if (world_rank == 0)
    {
        // TODO: PI result
		pi_result = 4 * count / (double) tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
