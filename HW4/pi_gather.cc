#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <random>

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
	lld count = 0;

	int rdn = rd();
	for(toss = 0; toss < iterations; toss++) {
		unsigned xy = xorshift( rdn );
		unsigned x = (xy & 0x7FFF0000) >> 16;
		unsigned y = xy & 0x00007FFF;
		count += ( (x*x + y*y) <= R*R);
	}
	return count;
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
	
	lld iterations = tosses / world_size;

	int tag = 0;
	int root = 0;
	lld count_arr[ world_size ];
	
    // TODO: use MPI_Gather
	lld count = count_toss( iterations );

	MPI_Gather(&count, 1, MPI_LONG_LONG, count_arr, 1, MPI_LONG_LONG, root, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // TODO: PI result
		int rank = 1;
		for( rank = 1; rank < world_size; rank++) {
			count += count_arr[ rank ];
		}
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
