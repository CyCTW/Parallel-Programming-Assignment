#include <iostream>
#include <mpi.h>
#include <string.h>
#include <immintrin.h>

using namespace std;

/* === fastIO === */
#define SIZE (1 << 21)
static char fr_buf[SIZE], fw_buf[SIZE];
char *frp1 = fr_buf, *frp2 = fr_buf, *fwp1 = fw_buf;
inline char Getchar() {
    if (frp1 != frp2) return *frp1++;
    frp2 = (frp1 = fr_buf) + fread(fr_buf, 1, SIZE, stdin);
    if (frp1 == frp2) return EOF;
    return *frp1++;
}
inline void flush() {
    fwrite(fw_buf, 1, fwp1 - fw_buf, stdout);
    fwp1 = fw_buf;
}
inline void Putchar(char c) {
    if (fwp1 - fw_buf == SIZE) flush();
    return *fwp1++ = c, void();
}
inline void print(int x) {
    if (x > 9) print(x / 10);
    Putchar(x % 10 + '0');
}
template <typename T>
inline void read(T &x) {
    x = 0; char s = Getchar();
    while (s < '0' || s > '9' ) s = Getchar();
    while (s >= '0' && s <= '9')
        x = x * 10 + s - '0', s = Getchar();
}
/* === fastIO === */
void print_matrix(int m, int l, const int* mat) {
    cout << "Print Matrix with size: " << m << " * " << l << '\n';
    for(int i=0; i<m; i++) {
        for(int j=0; j<l; j++) {
            cout << mat[i * l + j] << " \n"[j + 1 == l];
        }
    }

}

void sequential_mm(int n, int m, int l, const int* a_mat, const int* b_mat, int* c_mat) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<l; j++) {
            for (int k=0; k<m; k++) {
                c_mat[i * l + j] += a_mat[i * m + k] * b_mat[k * l + j];
            }
            cout << c_mat[i * l + j] << " \n"[j + 1 == l];
        }
    }
}

extern void construct_matrices(
    int *n_ptr, int *m_ptr, int *l_ptr,
    int **a_mat_ptr, int **b_mat_ptr )
{
    // ios_base::sync_with_stdio(false);
    // cin.tie(0);
    
    int world_size, world_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // sequential code: cin variable -> malloc matrix -> cin matrix
	
    int n, m, l, tag = 0, src = 0;
    if ( world_rank == 0) {

        // read 
		read<int>(*n_ptr);
		read<int>(*m_ptr);
		read<int>(*l_ptr);
//        cin >> *n_ptr >> *m_ptr >> *l_ptr;
    } 
    MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int a_size = (*n_ptr) * (*m_ptr);
    int b_size = (*m_ptr) * (*l_ptr);
    (*a_mat_ptr) = (int*) malloc( a_size * sizeof(int));
    (*b_mat_ptr) = (int*) malloc( b_size * sizeof(int));

    if (world_rank == 0) {
        // cout << "n: " << *n_ptr << ", m: " << *m_ptr << ", l: " << *l_ptr << '\n';
        // read matrix
        for(int i=0; i < (*n_ptr); i++) {
            for(int j=0; j < (*m_ptr); j++) {
                read<int>((*a_mat_ptr)[ i * (*m_ptr) + j]);
                // cin >> (*a_mat_ptr)[ i * (*m_ptr) + j] ;
            }
        }

        for(int i=0; i < (*m_ptr); i++) {
            for(int j=0; j < (*l_ptr); j++) {
                read<int>((*b_mat_ptr)[ i * (*l_ptr) + j ]);
                // cin >> (*b_mat_ptr)[ i * (*l_ptr) + j ];
            }
        }
    } 
    MPI_Bcast(*a_mat_ptr, a_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(*b_mat_ptr, b_size, MPI_INT, 0, MPI_COMM_WORLD);
    
}



void cal(int start, int end, int m, int l, const int* a_mat, const int* b_mat, int* c_mat) {
    
    for (int i = start; i < end; i++) {
        // for (int j = 0; j < l; j++) {
        int j=0;
        for (; j<=(l-8); j+=8) {
            const __m256i ones = _mm256_set1_epi32(0xFFFFFFFF);

            __m256i c0 = {0, 0, 0, 0};

            for (int k = 0; k < m; k++) {
                c0 = _mm256_add_epi32(
                    c0, 
                    _mm256_mullo_epi32(
                        _mm256_set1_epi32( *(a_mat + (i * m) + k) ),
                        _mm256_maskload_epi32( (b_mat + (k * l) + j), ones)
                    )
                );
                // equal to : c_mat[i][j] += a_mat[i * m + k] * b_mat[k * l + j];
                // c_mat[ (i-start) * l + j] += a_mat[i * m + k] * b_mat[k * l + j];
            }
            _mm256_maskstore_epi32( (c_mat + (i-start)*l + j), ones, c0);
        }
        if (l % 8) {
            
            for(; j<l; j++) {
                for(int k = 0; k < m; k++) {
                    c_mat[ (i-start) * l + j] += a_mat[i * m + k] * b_mat[k * l + j];
                }
            }
        }
    }
}

extern void matrix_multiply(
    const int n, const int m, const int l,
    const int *a_mat, const int *b_mat)
{
    // cout << "NML: " << n << ' ' << m << ' ' << l << '\n';
    
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
    for(int i=0; i<world_rank; i++)
        start+=offset[i];
    
    int oseg = n / world_size;
    int seg = n / world_size + offset[world_rank];    
    int end = start + seg;
    // cout << "Node " << world_rank << '\n';
    
    // cout << "Start row: " << start << ", End row: " << end << '\n';

    /*
    1 1 1 1
    1 1 1 2
    1 1 1 3
    1 1 1 4

    2 2 2 2
    2 2 2 3
    2 2 2 4
    2 2 2 5 
    3 3 3 2
    */
    int c_mat[ ( seg*l) ];

    memset(c_mat, 0, sizeof(c_mat));
    const int c_size = seg*l;
        
    // parallel compute 
    cal(start, end, m, l, a_mat, b_mat, c_mat);

    // collect data to node 0

    // int total_c_mat[ (n*l) ];
    // memset(total_c_mat, 0, sizeof(total_c_mat));

	// MPI_Gather(&c_mat, c_size, MPI_INT, &total_c_mat, c_size, MPI_INT, 0, MPI_COMM_WORLD);
    int tag = 0;
    
    if (world_rank == 0) {
        for(int j=0; j<c_size; j++) {
            if (j%l==0 && j!=0) Putchar('\n'); // cout << '\n';
            else if (j!=0) Putchar(' '); // cout << ' ';
            print(c_mat[j]); // cout << c_mat[j] ;
        }

        for(int i=1; i<world_size; i++) {
            int nodesiz = (oseg + offset[i])*l;
            int tmp[nodesiz];

            MPI_Recv(&tmp, nodesiz, MPI_INT, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for(int j=0; j<nodesiz; j++) {
                if (j % l == 0) Putchar('\n'); //cout << '\n';
                else Putchar(' '); //cout << ' ';
                print(tmp[j]); // cout << tmp[j];
            }
        }
		Putchar('\n');
        // cout << '\n';
		flush();
        // memset(total_c_mat, 0, sizeof(total_c_mat));
        // cout << "\n\n";
        // sequential_mm(n, m, l, a_mat, b_mat, total_c_mat);
    } else {
        MPI_Send(&c_mat, c_size, MPI_INT, 0, tag, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
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
