#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<pthread.h>
#include<string.h>
#include<unistd.h>

typedef long long lld;
lld total_num_of_tosses;
int thread_num;
lld in_circle[10] = { 0 };
int part = 0;

// thread exec function
void* sub_func(void* arg) {
	lld num_of_tosses = total_num_of_tosses / thread_num;
	int n = part++;

	for(lld toss = 0; toss<num_of_tosses; toss++) {
		double x = 2.0 * rand() / double(RAND_MAX);
		double y = 2.0 * rand() / double(RAND_MAX);

		// rescale range from 0~2 to -1~1
		x -= 1.0;
		y -= 1.0;
		if (x > 1.0 || y > 1.0) {
			printf("ERROR!\n\n");
			break;
		}

		// calculate distance from 0,0
		double dis = x*x+y*y;
		if (dis <= 1) {
			in_circle[n]++;
		}
	}
	printf("num, in_circle, num_of_tosses: %d, %lld %lld\n", n, in_circle[n], num_of_tosses);
	// pthread_exit((void *) &in_circle);
}


int main(int argc, char** argv) {
	clock_t begin = clock();
	srand( time(NULL) );

	thread_num = atoi(argv[1]);
	total_num_of_tosses = atoi(argv[2]);
	// memset(in_circle, 0, sizeof(in_circle));

	pthread_t pt[thread_num];
	for(int i=0; i<thread_num; i++) {
		pthread_create(&pt[i], NULL, sub_func, NULL);
	}
	lld total_in_circle = 0;
	for(int i=0; i<thread_num; i++) {
		void *res;
		pthread_join(pt[i], &res);
	}
	for(int i=0; i<thread_num; i++) {
		total_in_circle += in_circle[i];
	}
	double pi_ = 4 * total_in_circle / (double)total_num_of_tosses ;

	clock_t end = clock();

	printf("Estimated PI = %lf\n", pi_);
	printf("Calculation time = %.3f sec.\n", (double)(end-begin) / CLOCKS_PER_SEC);
	printf("RAND_MAX = %d\n", RAND_MAX);
}
