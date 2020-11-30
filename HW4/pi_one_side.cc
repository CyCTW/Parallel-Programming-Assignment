#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <random>
#include <string.h>

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

    MPI_Win win;

    // TODO: MPI init
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	
	lld iterations = tosses / world_size;
	int tag = 0;
	
	lld total_count;

    if (world_rank == 0)
    {
        // Master
		lld *count_arr;
		MPI_Alloc_mem(world_size * sizeof(lld), MPI_INFO_NULL, &count_arr);

		lld count = count_toss( iterations );
		
		MPI_Win_create(count_arr, world_size * sizeof(lld), sizeof(lld), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
		lld ready = 0;
		int i;
		int cnt = 0;
		while (ready == 0)
		{
			ready = 1;
			MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
			
			for(i = 1; (i < world_size) && (ready==0); i++) {
				ready = count_arr[ i ];
			}

			MPI_Win_unlock(0, win);
		}
		printf("All nodes finished\n");
		total_count = count;
		
		for(i = 1; i < world_size; i++) {
			total_count += count_arr[ i ];
		}
    }
    else
    {
        // Workers
		lld count = count_toss( iterations );

		// worker processes don't expose memory in the window
		MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
		MPI_Put(&count, 1, MPI_LONG_LONG, 0, world_rank, 1, MPI_LONG_LONG, win);
		MPI_Win_unlock(0, win);
		printf("Rank %d Finished\n", world_rank);
    }

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
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
