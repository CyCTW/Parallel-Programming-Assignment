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
	lld total_count = 0;
	lld count;

	lld src_count[ world_size + 1 ];
    if (world_rank > 0)
    {
        // TODO: MPI workers
		count = count_toss( iterations );
		int dest_rank = 0;
		MPI_Request req;
		MPI_Isend(&count, 1, MPI_LONG_LONG, dest_rank, tag, MPI_COMM_WORLD, &req);
		MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request requests[ world_size - 1];
		int rank;
		for(rank = 1; rank < world_size; rank++) {
			MPI_Irecv(&src_count[rank], 1, MPI_LONG_LONG, rank, tag, MPI_COMM_WORLD, &requests[rank-1]);
		}
		total_count = count_toss( iterations );
        MPI_Waitall( world_size - 1, requests, MPI_STATUS_IGNORE);
    }

    if (world_rank == 0)
    {
        // TODO: PI result
		int rank;
		for(rank = 1; rank < world_size; rank++) {
			total_count += src_count[rank];
		}
		pi_result = 4 * total_count / (double) tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
