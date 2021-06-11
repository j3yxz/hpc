/****************************************************************************
 *
 * skyline.c
 *
 * Serial implementaiton of the skyline operator
 *
 * Copyright (C) 2020 Moreno Marzolla
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * --------------------------------------------------------------------------
 *
 * Questo programma calcola lo skyline di un insieme di punti in D
 * dimensioni letti da standard input. Per una descrizione completa
 * si veda la specifica del progetto sul sito del corso:
 *
 * https://www.moreno.marzolla.name/teaching/HPC/
 *
 * Per compilare:
 *
 * gcc -D_XOPEN_SOURCE=600 -std=c99 -Wall -Wpedantic -O2 skyline.c -o skyline
 *
 * (il flag -D_XOPEN_SOURCE=600 e' superfluo perche' viene settato
 * nell'header "hpc.h", ma definirlo tramite la riga di comando fa si'
 * che il programma compili correttamente anche se non si include
 * "hpc.h", o non lo si includa come primo file).
 *
 * Per eseguire il programma:
 *
 * ./skyline < input > output
 *
 ****************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLKDIM 1024
typedef struct {
    float *P;   /* coordinates P[i][j] of point i               */
    int N;      /* Number of points (rows of matrix P)          */
    int D;      /* Number of dimensions (columns of matrix P)   */
} points_t;

/**
 * Read input from stdin. Input format is:
 *
 * d [other ignored stuff]
 * N
 * p0,0 p0,1 ... p0,d-1
 * p1,0 p1,1 ... p1,d-1
 * ...
 * pn-1,0 pn-1,1 ... pn-1,d-1
 *
 */
void read_input( points_t *points )
{
    char buf[1024];
    int N, D, i, k;
    float *P;

    if (1 != scanf("%d", &D)) {
        fprintf(stderr, "FATAL: can not read the dimension\n");
        exit(EXIT_FAILURE);
    }
    assert(D >= 2);
    if (NULL == fgets(buf, sizeof(buf), stdin)) { /* ignore rest of the line */
        fprintf(stderr, "FATAL: can not read the first line\n");
        exit(EXIT_FAILURE);
    }
    if (1 != scanf("%d", &N)) {
        fprintf(stderr, "FATAL: can not read the number of points\n");
        exit(EXIT_FAILURE);
    }
    P = (float*)malloc( D * N * sizeof(*P) );
    assert(P);
    for (i=0; i<N; i++) {
        for (k=0; k<D; k++) {
            if (1 != scanf("%f", &(P[i*D + k]))) {
                fprintf(stderr, "FATAL: failed to get coordinate %d of point %d\n", k, i);
                exit(EXIT_FAILURE);
            }
        }
    }
    points->P = P;
    points->N = N;
    points->D = D;
}

void free_points( points_t* points )
{
    free(points->P);
    points->P = NULL;
    points->N = points->D = -1;
}

/* Returns 1 iff |p| dominates |q| */
int dominates( const float * p, const float * q, int D )
{
    int k;

    /* The following loop could be merged, but the keep them separated
       for the sake of readability */
    for (k=0; k<D; k++) {
        if (p[k] < q[k]) {
            return 0;
        }
    }
    for (k=0; k<D; k++) {
        if (p[k] > q[k]) {
            return 1;
        }
    }
    return 0;
}

/**
 * Compute the skyline of |points|. At the end, s[i] == 1 iff point
 * |i| belongs to the skyline. This function returns the number r of
 * points in to the skyline. The caller is responsible for allocating
 * a suitably sized array |s|.
 */

__global__ void dominates_kernel(int p, int N, int D, float* d_i, int* d_s, float* d_P){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float s_p[];
    int k;
    if(tid >= N || tid == p || d_s[tid] == 0){
        return;
    } else {
        //dominates d_s[p] & d_s[tid]
        for (k=0; k<D; k++) {
            if (d_i[tid*D+k] < d_P[tid*D+k]) {
                return;
            }
        }
        for (k=0; k<D; k++) {
            if (d_i[tid*D+k] > d_P[tid*D+k]) {
                d_s[tid]=0;
                return;
            }
        }
    }
    return;
}
__global__ void kernel(int i, int N, int D, int *d_s, float *d_P){

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float p[];
    int k;
    float temp;



    if(threadIdx.x < D){
        /* per accedere in memoria globale una sola volta anziché BLKDIM (credo che dovrebbe ottimizzarlo il compilatore, ma per sicurezza lo lascio) */
        temp=d_P[i*D + threadIdx.x];
        for(k=0; k<32; k++) {
            p[threadIdx.x * blockDim.x + k] = temp;
           // p[tid]=-10;
            //d_i[tid]=temp;
        }
        //p[threadIdx.x + BLKDIM] = d_P[i + threadIdx.x];
    }
    //fin qui tutto
#if 1
    __syncthreads();
    if( tid >= N || tid == i ||d_s[tid] == 0){
        return;
    } else {
        //dominates d_s[p] & d_s[tid]
        
        for (k=0; k<D; k++) {
            if (p[blockDim.x * k + threadIdx.x] < d_P[tid*D+k]) {
                return;
            }
        }
        for (k=0; k<D; k++) {
            if (p[blockDim.x * k + threadIdx.x] > d_P[tid*D+k]) {

                d_s[tid]=0;
                return;
            }
        }
    }
    //d_i[tid] = p[threadIdx.x];
#endif
    //d_i[tid] = -3;


    return;
}
int skyline_cuda( const points_t *points, int *s )
{
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;
    const int nblocks=1+(N-1)/32;
    printf("nblock : %d\n", nblocks);
    int i, r = 0;
    
    float *h_i;
    h_i=(float*)calloc(N*D,sizeof(float));
    
   // for(int k=0; k<N*D; k++) h_i[k]=P[k%D];

    for (i=0; i<N; i++) {
        s[i] = 1;
    }
    int *d_s;
    float *d_P;
    float *d_i;
    float *h_test=(float*)calloc(32*2,sizeof(float));
    float *test;
    float *t;
    float *h_t=(float*)calloc(96,sizeof(float));
    
    cudaSafeCall( cudaMalloc( (void**)&d_s, N*sizeof(int)) );
    cudaSafeCall( cudaMalloc( (void**)&d_P, N*D*sizeof(float)) );
    cudaSafeCall( cudaMalloc( (void**)&d_i, N*D*sizeof(float)) );
    cudaSafeCall( cudaMalloc( (void**)&test, 32*2*sizeof(float)) );
    cudaSafeCall( cudaMalloc( (void**)&t, 96*sizeof(float)) );
    cudaSafeCall( cudaMemcpy( d_s, s, N*sizeof(int), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy( d_P, P, N*D*sizeof(float), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy( d_i, h_i, N*D*sizeof(float), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy( test, h_test, 32*2*sizeof(float), cudaMemcpyHostToDevice) );

    
    for (i=0; i<N; i++) {
        if ( s[i] ) {
            //for(int k=0; k<N*D; k++) h_i[k]=P[i*D+k%D];
            //cudaMemcpy( d_i, h_i, N*D*sizeof(float), cudaMemcpyHostToDevice);
            kernel<<<nblocks,32,D*32*sizeof(float)>>>(i, N, D, d_s, d_P);
            //printf("porco dio\n");
            cudaSafeCall ( cudaMemcpy( h_i, d_i, D*N*sizeof(float), cudaMemcpyDeviceToHost) );
            //cudaCheckError();
            cudaSafeCall( cudaMemcpy( s, d_s, N*sizeof(int), cudaMemcpyDeviceToHost) );
            cudaSafeCall( cudaMemcpy( d_s, s, N*sizeof(int), cudaMemcpyHostToDevice) );

            //printf("\n");
            //printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", h_i[0], h_i[1], h_i[2],h_i[3],h_i[4],h_i[5],h_i[6],h_i[7],h_i[8],h_i[9],h_i[10],h_i[11],h_i[12],h_i[13],h_i[14],h_i[15],h_i[16],h_i[33]);
            //printf("\np");
            //for(int lel=0;lel<96; lel++) printf(" %f",h_t[lel]);

            //for(int lel=0;lel<64; lel+=2) printf("tid=[%d] %f %f      \n",lel/2,h_test[lel], h_test[lel+1] );
            //printf("\n");
            //printf("%d\n", i );

        }
    }

    
    //kernel<<<4,32,2*D*sizeof(float)>>>();
    for(i=0;i<N;i++){
        if(s[i]) r++;
    }
    return r;
}

int skyline( const points_t *points, int *s )
{
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;
    int i, j, r = N;

    for (i=0; i<N; i++) {
        s[i] = 1;
    }

    for (i=0; i<N; i++) {
        if ( s[i] ) {
            for (j=0; j<N; j++) {
                if ( s[j] && dominates( &(P[i*D]), &(P[j*D]), D ) ) {
                    s[j] = 0;
                    r--;
                }
            }
        }
    }
    return r;
}

/**
 * Print the coordinates of points belonging to the skyline |s| to
 * standard ouptut. s[i] == 1 iff point i belongs to the skyline.  The
 * output format is the same as the input format, so that this program
 * can process its own output.
 */
void print_skyline( const points_t* points, const int *s, int r )
{
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;
    int i, k;

    printf("%d\n", D);
    printf("%d\n", r);
    for (i=0; i<N; i++) {
        if ( s[i] ) {
            for (k=0; k<D; k++) {
                printf("%f ", P[i*D + k]);
            }
            printf("\n");
        }
    }
}

int main( int argc, char* argv[] )
{
    points_t points;

    if (argc != 1) {
        fprintf(stderr, "Usage: %s < input_file > output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    read_input(&points);
    int *s = (int*)malloc(points.N * sizeof(*s));
    assert(s);
    const double tstart = hpc_gettime();
    const int r = skyline(&points, s);
    const double elapsed = hpc_gettime() - tstart;
    //print_skyline(&points, s, r);

    fprintf(stderr,
            "\n\t%d points\n\t%d dimensione\n\t%d points in skyline\n\nExecution time %f seconds\n",
            points.N, points.D, r, elapsed);

    const double cuda_start = hpc_gettime();
    const int cuda_r = skyline_cuda(&points, s);
    const double cuda_elapsed = hpc_gettime() - cuda_start;
    //print_skyline(&points, s, r);

    fprintf(stderr,
            "\n\t%d points\n\t%d dimensione\n\t%d points in skyline\n\nExecution time %f seconds\n",
            points.N, points.D, cuda_r, cuda_elapsed);
    free_points(&points);
    free(s);
    return EXIT_SUCCESS;
}
