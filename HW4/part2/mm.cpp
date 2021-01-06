#include <iostream>
#include <mpi.h>
#include <string.h>
#include <immintrin.h>

using namespace std;

#define BUFSIZE (1 << 22)
static char fr_buf[ BUFSIZE ];
static char fw_buf[ BUFSIZE ];
char *frp1 = fr_buf, *frp2 = fr_buf, *fwp1 = fw_buf;

inline char Getchar() {
    if (frp1 != frp2) return *frp1++;
    frp1 = fr_buf;
    frp2 = (frp1) + fread(fr_buf, 1, BUFSIZE, stdin);

    if (frp1 == frp2) return EOF;
    else return *frp1++;
}
inline void flush() {
    fwrite(fw_buf, 1, fwp1 - fw_buf, stdout);
    fwp1 = fw_buf;
}
inline void Putchar(char c) {
    if (fwp1 - fw_buf == BUFSIZE) flush();
    *fwp1++ = c;
}
inline void PutNum(int x) {
    if (x > 9) PutNum(x / 10);
    Putchar(x % 10 + '0');
}
template <typename T>
inline void read(T &x) {
    x = 0; 
    char s = Getchar();
    while (s < '0' || s > '9' ) s = Getchar();
    while (s >= '0' && s <= '9')
        x = x * 10 + s - '0', s = Getchar();
}


extern void construct_matrices(
    int *n_ptr, int *m_ptr, int *l_ptr,
    int **a_mat_ptr, int **b_mat_ptr )
{
    int world_size, world_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // sequential code: cin variable -> malloc matrix -> cin matrix
	
    int tag = 0, src = 0;
    if ( world_rank == 0) {

        // read 
		read<int>(*n_ptr);
		read<int>(*m_ptr);
		read<int>(*l_ptr);
    } 
    int padding = ( 16 - ((*m_ptr) % 16) );
    int m = (*m_ptr);
    (*m_ptr) += padding;

    MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);


    int a_size = (*n_ptr) * (*m_ptr);
    int b_size = (*m_ptr) * (*l_ptr);
    (*a_mat_ptr) = new int[ a_size ];
    (*b_mat_ptr) = new int[ b_size ];

    if (world_rank == 0) {
        // read matrix
        for(int i=0; i < (*n_ptr); i++) {
            for(int j=0; j < (m); j++) {
                read<short>( ( (short*)(*a_mat_ptr) )[ i * (*m_ptr) + j] );
            }
            // pad to fit 16
            for(int j = (m); j < (m + padding); j++) {
                ( (short*)(*a_mat_ptr) )[ i * (*m_ptr) + j] = 0;
            }
        }

        for(int i=0; i < (m); i++) {
            for(int j=0; j < (*l_ptr); j++) {
                read<short>( ( (short*)(*b_mat_ptr) )[ j * (*m_ptr) + i ]);
                // Directly Transpose Matrix and Store short in int
            }
        }
        for(int i=(m); i<(m + padding); i++) {
            for(int j=0; j<(*l_ptr); j++) {
                ( (short*)(*b_mat_ptr) )[ j * (*m_ptr) + i ] = 0;
            }
        }
    } 
    MPI_Bcast(*a_mat_ptr, a_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(*b_mat_ptr, b_size, MPI_INT, 0, MPI_COMM_WORLD);
    
}



void cal(int start, int end, int m, int l, const int* a_mat, const int* b_mat, int* c_mat) {

    const __m256i ones = _mm256_set1_epi32(0xFFFFFFFF);
    const __m256i zeros = _mm256_set1_epi32(0);

    for (int i = start; i < end; i++) {
        for(int j = 0; j < l; j++) {
            __m256i summ = _mm256_set1_epi32(0);
            /* -- SIMD Intrinsics -- */
            int k;
            for (k = 0; k <= (m-16); k += 16) {
                __m256i c0 = _mm256_mullo_epi16(
                        // _mm256_set1_epi32( *(a_mat + (i * m) + k) ),
                        _mm256_maskload_epi32( (const int*)( ( (short*)a_mat ) + (i * m) + k), ones ),
                        _mm256_maskload_epi32( (const int*)( ( (short*)b_mat ) + (j * m) + k), ones )
                    );

                __m256i lo = _mm256_unpacklo_epi16(c0, zeros);
                __m256i hi = _mm256_unpackhi_epi16(c0, zeros);
                summ = _mm256_add_epi32( summ, lo );
                summ = _mm256_add_epi32( summ, hi );

            }
            // horizonal add 4 to 1
            summ = _mm256_hadd_epi32(summ, summ);
            summ = _mm256_hadd_epi32(summ, summ);

            int *ptr1 = (int*)&summ;
            int csum = ptr1[0] + ptr1[4];

            c_mat[ (i-start) * l + j] = csum;
            /* -- SIMD Intrinsics -- */

        }
    }
}

extern void matrix_multiply(
    const int n, const int m, const int l,
    const int *a_mat, const int *b_mat)
{
    int world_size, world_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    int remain = n % world_size;
    int offset[world_size];

    for(int i = 0; i < world_size; i++) {
        if (i < remain)
            offset[i] = 1;
        else
            offset[i] = 0;
    }

    int start = world_rank * (n / world_size);
    for(int i=0; i < world_rank; i++)
        start += offset[i];
    
    int origin_seg = n / world_size;
    int seg = n / world_size + offset[world_rank];    
    int end = start + seg;

    int *c_mat = new int[ seg*l ];

    const int c_size = seg*l;
        
    // parallel compute 
    cal(start, end, m, l, a_mat, b_mat, c_mat);
    // collect data to node 0
    int tag = 0;
    
    if (world_rank == 0) {
        for(int j=0; j < c_size; j++) {
            PutNum(c_mat[j]);
            Putchar(" \n"[ ((j+1)%l) == 0  ]); 
        }

        for(int i=1; i<world_size; i++) {
            int nodesiz = (origin_seg + offset[i]) * l;
            int* partialc = new int[ nodesiz ]; 
            MPI_Recv(partialc, nodesiz, MPI_INT, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for(int j=0; j < nodesiz; j++) {
                PutNum(partialc[j]); 
                Putchar(" \n"[ (j+1)%l == 0] );
            }
            delete []partialc;
        }
		flush();
    } else {
        MPI_Send(c_mat, c_size, MPI_INT, 0, tag, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    delete []c_mat;
}

extern void destruct_matrices(int *a_mat, int *b_mat)
{
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        delete []a_mat;
        delete []b_mat;
    }
}
