#include<stdio.h>
#include<stdlib.h>
#include<time.h>

typedef long long lld;

int main() {

	srand( time(NULL) );
	lld num_of_tosses = 1e8;
	lld in_circle = 0;

	lld r = 1, l = -1;

	for(lld toss = 0; toss<num_of_tosses; toss++) {
		double x = 2.0 * rand() / double(RAND_MAX);
		double y = 2.0 * rand() / double(RAND_MAX);
		x -= 1.0;
		y -= 1.0;
		if (x > 1.0 || y > 1.0) {
			printf("ERROR!\n\n");
			break;
		}
		double dis = x*x+y*y;
		if (dis <= 1) {
			in_circle++;
		}
	}
	double pi_ = 4 * in_circle / (double)num_of_tosses ;
	printf("Estimated PI = %lf\n", pi_);
	printf("RAND_MAX = %d\n", RAND_MAX);
}
