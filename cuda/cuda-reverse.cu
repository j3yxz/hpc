/* */
/****************************************************************************
 *
 * cuda-reverse.cu - Array reversal with CUDA
 *
 * Copyright (C) 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 * Last modified on 2020-05-23 by Moreno Marzolla
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * ---------------------------------------------------------------------------
 *
 * Compile with:
 * nvcc cuda-reverse.cu -o cuda-reverse
 *
 * Run with:
 * ./cuda-reverse [n]
 *
 * Example:
 * ./cuda-reverse
 *
 ****************************************************************************/
#include "hpc.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#define BLKDIM 1024
/* Reverse in[] into out[].
   [TODO] This function should be rewritten as a kernel. */
void reverse( int *in, int *out, int n )
{
    int i;
    for (i=0; i<n; i++) {
        out[n - 1 - i] = in[i];
    }
}
//kernel versione of revers() there must be threads >= n
__global__ void reverse_kernel(int *in, int *out, int n){

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < n){
        out[tid] = in[n-tid-1];
    }



}

/* In-place reversal of in[] into itself.
   [TODO] This function should be rewritten as a kernel. */
void inplace_reverse( int *in, int n )
{
    int i = 0, j = n-1;
    while (i < j) {
        int tmp = in[j];
        in[j] = in[i];
        in[i] = tmp;
        j--;
        i++;
    }
}
//kernel version of inplace_revers() there must be threads >= n/2
__global__ void inplace_reverse_kernel(int *in, int n ){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tmp;

    if(tid < (n+1)/2){
        tmp=in[tid];
        in[tid] = in[n-tid-1];
        in[n-tid-1]=tmp;
    }

}
void fill( int *x, int n )
{
    int i;
    for (i=0; i<n; i++) {
        x[i] = i;
    }
}

int check( int *x, int n )
{
    int i;
    for (i=0; i<n; i++) {
        if (x[i] != n - 1 - i) {
            fprintf(stderr, "Test FAILED: x[%d]=%d, expected %d\n", i, x[i], n-1-i);
            return 0;
        }
    }
    printf("Test OK\n");
    return 1;
}

int main( int argc, char* argv[] )
{
    int *h_in, *h_out;  /* host copy of array in[] and out[] */
    int *d_in, *d_out; /* device copy of arrays*/
    int n = 1024*1024;
    const int max_len = 512*1024*1024;
    
    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > max_len ) {
        fprintf(stderr, "FATAL: the maximum length is %d\n", max_len);
        return EXIT_FAILURE;
    }
    
    const size_t size = n * sizeof(*h_in);

    /* Allocate host copy of in[] and out[] */
    h_in = (int*)malloc(size); assert(h_in);
    fill(h_in, n);
    h_out = (int*)calloc(1,size); assert(h_out);
    cudaMalloc( (void**)&d_in, size);
    cudaMalloc( (void**)&d_out, size);
    cudaMemcpy( d_in, h_in, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_out, h_out, n*sizeof(int), cudaMemcpyHostToDevice);
    
    int nblocks_k1 = (n + BLKDIM-1)/(BLKDIM);
    int nblocks_k2=(nblocks_k1+1)/2;
    /* Reverse
       [TODO] Rewrite as a kernel launch. 
    printf("Reverse %d elements... ", n);
    reverse(h_in, h_out, n);    
    check(h_out, n);
    */
    reverse_kernel<<<nblocks_k1,BLKDIM>>>( d_in, d_out, n);
    cudaMemcpy( h_out, d_out, n*sizeof(int), cudaMemcpyDeviceToHost);
    check(h_out, n);
    inplace_reverse_kernel<<<nblocks_k2,BLKDIM>>>( d_in , n);
    cudaMemcpy( h_in, d_in, n*sizeof(int), cudaMemcpyDeviceToHost);
    check(h_in, n);
    /* In-place reverse 
       [TODO] Rewrite as a kernel launch. 
    printf("In-place reverse %d elements... ", n);
    inplace_reverse(h_in, n);
    */    
    
    /* Cleanup */
    free(h_in); free(h_out); cudaFree(d_out); cudaFree(d_in);
    return EXIT_SUCCESS;
}
