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
lld total_num_of_tosses;
int thread_num;
lld in_circle[10] = { 0 };
int part = 0;


lld rand_max = 2147483647;
// std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(0,1);
int r = (1<<15);
// std::uniform_int_distribution<unsigned> dist2(0, (r));
using namespace std;



// thread exec function
void* sub_func(void* arg) {

std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<unsigned> dist2(0, (r));

	int num_of_tosses = total_num_of_tosses / thread_num;
	int n = part++;

	for(int toss = 0; toss<num_of_tosses; toss++) {
		// float x = dist(gen) ;
		// float y = dist(gen) ;
		unsigned x = dist2(gen);
		unsigned y = dist2(gen);

		// calculate distance from 0,0
		// area ratio => r*r : r*r*pi/4 = num_of_tosses : tosses
		// float dis = x*x+y*y;
		unsigned dis = x*x+y*y;
		// cout << dis << ' ' << r*r << '\n';
		// if (dis <= 1) {
		if (dis <= r*r) {
			in_circle[n]++;
		}
	}
	cout << "Thread_num: " << n << "in_circle: " << in_circle[n] << "num_of_tosses: " << num_of_tosses << '\n';
	return NULL;
}

int gogo(int &state) {
	int x = state;
	x ^= (x << 13);
	x ^= (x >> 17);
	x ^= (x << 5);
	return state = x;
}

void* sub_func_SIMD(void *arg) {

	random_device rd;
	// std::mt19937 gen(rd());
	// std::uniform_int_distribution<unsigned> dist2(0, (r));

	int state = rd();
	int num_of_tosses = total_num_of_tosses / thread_num;
	int n = part++;
	int vec_width = 8;
	
	// range: (0~2^15)
	int rand_x[8] = {1, 3, 3, 2, 32, 55, 4, 3};
	int rand_y[8] = {1, 3, 3, 2, 32, 55, 4, 3};
	
	
	// 32 bit integer
	__m256i res = _mm256_setzero_si256();
	const __m256i r_square = _mm256_set1_epi32(r*r+1);
	const __m256i ones = _mm256_set1_epi32(0xFFFFFFFF);
	const __m256i one = _mm256_set1_epi32(1);


	for(int toss = 0; toss < num_of_tosses; toss += vec_width) {
		// load random x, y
		for(int i=0; i<8; i++) {
			rand_x[i] = gogo(state) % (r + 1); //dist2(gen);
			rand_y[i] = gogo(state) % (r + 1); //dist2(gen);
		}

		const __m256i x = _mm256_maskload_epi32( rand_x, ones );
		const __m256i y = _mm256_maskload_epi32( rand_y, ones );
		
		// calculate distance: x*x+y*y
		const __m256i dis = _mm256_add_epi32( _mm256_mullo_epi32(x, x), _mm256_mullo_epi32(y, y) );
		
		// if ( r+1 > dis)
		__m256i mk_int = _mm256_cmpgt_epi32(r_square, dis); 
		
		mk_int = _mm256_and_si256(mk_int, one);
		// res+=1
		res = _mm256_add_epi32(mk_int, res);
		// int* pp = (int*)&res;

		// for(int i=0; i<8; i++) {
		// 	cout << pp[i] << ' ';
		// }
		// cout << '\n';
	}


	// store output
	// 8 32-bit int in res, add them up
	int total = 0;
	// for(int idx=0; idx<8; idx++) {
	// 	// const int xx = idx;
	// 	total += p[idx];
	// 	// int tmp = _mm256_extract_epi32(res, xx);
	// 	// total += tmp;		
	// }
	int* p = (int*)&res;
	res = _mm256_hadd_epi32(res, res);
	res = _mm256_hadd_epi32(res, res);
	total = p[0] + p[4];
	// const __m128i low  = _mm256_castsi256_si128( res );
	// const __m128i high = _mm256_extractf128_si256( res, 1);
	// const __m128i r4 = _mm_add_epi32(low, high);
	// remain 4 int in res
	// const __m128 r2 = _mm_add_ps (r4, _mm_movehl_ps( r4, r4) ) ;
	// const __m128 r1 = _mm_add_ss (r2, _mm_movehdup_ps( r2) );
	// cout << " Total: " << total << '\n';
	in_circle[n] = total;
	return NULL;
	
}

int main(int argc, char** argv) {

	thread_num = atoi(argv[1]);
	// total_num_of_tosses = atoi(argv[2]);
	total_num_of_tosses = 1e8;

	pthread_t pt[thread_num];
	for(int i=0; i<thread_num; i++) {
		pthread_create(&pt[i], NULL, sub_func_SIMD, NULL);
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
}
