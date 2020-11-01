#include<iostream>
#include<time.h>
#include<pthread.h>
#include<string.h>
#include<unistd.h>
#include<random>
#include<chrono>

// AVX SIMD intrinsics
#include<immintrin.h>

typedef long long lld;
using namespace std;

lld total_num_of_tosses;
int thread_num;

lld in_circle[40000] = { 0 };

int R = (1<<15)-1;

// thread exec function
void* sub_func(void* arg) {

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<unsigned> dist2(0, (R));

	lld num_of_tosses = total_num_of_tosses / thread_num;
	int n = *(int*)arg;

	for(lld toss = 0; toss<num_of_tosses; toss++) {
		// float x = dist(gen) ;
		// float y = dist(gen) ;
		unsigned x = dist2(gen);
		unsigned y = dist2(gen);

		// calculate distance from 0,0
		// area ratio => R*R : R*R*pi/4 = num_of_tosses : tosses
		// float dis = x*x+y*y;
		unsigned dis = x*x+y*y;
		// cout << dis << ' ' << R*R << '\n';
		// if (dis <= 1) {
		if (dis <= R*R) {
			in_circle[n]++;
		}
	}
	// cout << "Thread_num: " << n << "in_circle: " << in_circle[n] << "num_of_tosses: " << num_of_tosses << '\n';
	return NULL;
}
void print(__m256i p) {
	int* pp = (int*)&p;

	for(int i=0; i<8; i++) {
		cout << pp[i] << ' ';
	}
	cout << '\n';
}

int xorshift(int &state) {
	int x = state;
	x ^= (x << 13);
	x ^= (x >> 17);
	x ^= (x << 5);
	return state = x;
}

__m256i _mm256_mod(__m256i x, __m256i y) {
	// R = 2^15-1
	// x mod y = x & 11111111111
	return _mm256_and_si256(x, y);
}

__m256i xorshift_parallel(__m256i &stateP) {
	
	__m256i x = stateP;
	x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
	x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
	x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
	stateP = x;
	
	return _mm256_mod(x, _mm256_set1_epi32( R ) );
}

void sumup(__m256i &res, lld &total) {
	res = _mm256_hadd_epi32(res, res);
	res = _mm256_hadd_epi32(res, res);
	int* p = (int*)&res;

	total += lld(p[0]) + lld(p[4]);
	res = _mm256_setzero_si256();
}

void* sub_func_SIMD(void *arg) {

	random_device rd;	
	// std::mt19937 gen(rd());
	// std::uniform_int_distribution<unsigned> dist2(0, (R));


	lld num_of_tosses = total_num_of_tosses / thread_num;

	// int n = part++;
	int n = *(int *)arg;
	int vec_width = 8;

	
	// 32 bit integer
	__m256i res = _mm256_setzero_si256();
	const __m256i r_square = _mm256_set1_epi32(R*R+1);
	const __m256i ones = _mm256_set1_epi32(0xFFFFFFFF);
	const __m256i one = _mm256_set1_epi32(1);

	
	int int_state[8];
	for(int i = 0; i < 8; i++) {
		int_state[i] = rd();
	}

	__m256i state = _mm256_maskload_epi32( int_state, ones );
	
	lld total = 0;

	for(lld toss = 0; toss < num_of_tosses; toss += vec_width) {
		// load random x, y
		const __m256i x = xorshift_parallel(state);
		const __m256i y = xorshift_parallel(state);

		// calculate distance: x*x+y*y
		const __m256i dis = _mm256_add_epi32( _mm256_mullo_epi32(x, x), _mm256_mullo_epi32(y, y) );
		
		// if ( R*R+1 > dis)
		__m256i mk_int = _mm256_cmpgt_epi32(r_square, dis); 
		
		mk_int = _mm256_and_si256(mk_int, one);
		// res+=1
		res = _mm256_add_epi32(mk_int, res);

		// in case of int overflow
		if (toss % (1<<25) == 0) {
			sumup(res, total);
		}
	}

	// store output
	// 8 32-bit int in res, add them up
	sumup(res, total);

	in_circle[n] = total;
	// cout << "Thread " << n << " sum: " << total << '\n';
	return NULL;
}

int main(int argc, char** argv) {

	thread_num = atoi(argv[1]);
	total_num_of_tosses = atoll(argv[2]);

	// cal_counts = total_num_of_tosses % thread_num;

	// total_num_of_tosses = (total_num_of_tosses / lld(thread_num*8) + 1) * thread_num*8;

	cout << "Total num of tosses: " << total_num_of_tosses << '\n';

	pthread_t pt[thread_num];
	
	int thread_id[thread_num+10];
	memset(thread_id, 0, sizeof(thread_id));
	for(int i=0; i<thread_num; i++) {
		thread_id[i] = i;
	}


	for(int i = 0; i < thread_num; i++) {
		pthread_create(&pt[i], NULL, sub_func_SIMD, (void*)&thread_id[i]);
	}
	for(int i = 0; i < thread_num; i++) {
		pthread_join(pt[i], NULL);
	}
	
	lld total_in_circle = 0;
	for(int i = 0; i < thread_num; i++) {
		total_in_circle += in_circle[i];
	}

	double pi_ = 4 * total_in_circle / (double)total_num_of_tosses ;

	cout <<"Estimated PI = " << pi_ << '\n';
}
