#include<iostream>
#include<time.h>
#include<pthread.h>
#include<string.h>
#include<unistd.h>
#include<random>
#include<chrono>

typedef long long lld;
lld total_num_of_tosses;
int thread_num;
lld in_circle[10] = { 0 };
int part = 0;

lld rand_max = 2147483647;
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dist(-1,1);
using namespace std;
// thread exec function
void* sub_func(void* arg) {
	lld num_of_tosses = total_num_of_tosses / thread_num;
	int n = part++;

	for(lld toss = 0; toss<num_of_tosses; toss++) {
		double x = dist(gen) ;
		double y = dist(gen) ;
		// double x = 2.0 * rand() / double(RAND_MAX);
		// double y = 2.0 * rand() / double(RAND_MAX);

		// rescale range from 0~2 to -1~1

		// calculate distance from 0,0
		double dis = x*x+y*y;
		if (dis <= 1) {
			in_circle[n]++;
		}
	}
	cout << "Thread_num: " << n << "in_circle: " << in_circle[n] << "num_of_tosses: " << num_of_tosses << '\n';
}


int main(int argc, char** argv) {
	//srand( time(NULL) );


	thread_num = atoi(argv[1]);
	total_num_of_tosses = atoi(argv[2]);

	pthread_t pt[thread_num];
	for(int i=0; i<thread_num; i++) {
		pthread_create(&pt[i], NULL, sub_func, NULL);
	}
	for(int i=0; i<thread_num; i++) {
		pthread_join(pt[i], NULL);
	}
	
	lld total_in_circle = 0;
	for(int i=0; i<thread_num; i++) {
		total_in_circle += in_circle[i];
	}
	double pi_ = 4 * total_in_circle / (double)total_num_of_tosses ;

	cout <<"Estimated PI = " << pi_ << '\n';
	cout << RAND_MAX << '\n';
	cout << rand_max << '\n';
}
