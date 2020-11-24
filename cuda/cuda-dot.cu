/* */
/****************************************************************************
 *
 * cuda-dot.cu - Dot product with CUDA
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
 * nvcc cuda-dot.cu -o cuda-dot -lm
 *
 * Run with:
 * ./cuda-dot [len]
 *
 * Example:
 * ./cuda-dot
 *
 ****************************************************************************/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/***************/
#define BLKDIM 1024
__global__ void dot_kernel( double *x, double *y, int n, double *tmp ){

	const int tid = threadIdx.x;
	int i;
	double s=0.0;

	for ( i=tid; i<n; i+=BLKDIM ){
		s += x[i] * y[i]; 
	}
	tmp[tid] = s;
	__syncthreads();


	int active = (blockDim.x + 1)/2;
	

	while (active){
		
		if ( tid < active && active%2 == 0){
			tmp[tid] += tmp[tid + active];
		}	
        if ( tid < active -1 && active%2 != 0){
            tmp[tid] += tmp[tid + active];
        }   
        
		active/=2;
        __syncthreads();
	}
    if(0 == tid) tmp[0]+=tmp[1];
	
}

double dot( double *x, double *y, int n )
{
    /* [TODO] modify this function so that (part of) the dot product
       computation is executed on the GPU. **
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += x[i] * y[i];
    }
    return result;
    */
    double *d_x,*d_y,*d_tmp;
    double risultato=-1;
    const size_t size_xy = n*sizeof(double);
    const size_t size_tmp = BLKDIM*sizeof(double);


    cudaMalloc((void**)&d_x, size_xy);
	cudaMalloc((void**)&d_y, size_xy);
	cudaMalloc((void**)&d_tmp,size_tmp);
	cudaMemcpy(d_y, y, size_xy,cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, size_xy,cudaMemcpyHostToDevice);
	
    dot_kernel<<< 1,BLKDIM >>>(d_x, d_y, n, d_tmp);
	cudaCheckError();
	cudaMemcpy(&risultato,d_tmp,sizeof(double),cudaMemcpyDeviceToHost);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_tmp);
	return risultato;
}

void vec_init( double *x, double *y, int n )
{
    int i;
    const double tx[] = {1.0/64.0, 1.0/128.0, 1.0/256.0};
    const double ty[] = {1.0, 2.0, 4.0};
    const size_t arrlen = sizeof(tx)/sizeof(tx[0]);

    for (i=0; i<n; i++) {
        x[i] = tx[i % arrlen];
        y[i] = ty[i % arrlen];
    }
}

int main( int argc, char* argv[] ) 
{
    double *x, *y, result;
    int n = 1024*1024;
    const int max_len = 128 * n;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [len]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > max_len ) {
        fprintf(stderr, "FATAL: the maximum length is %d\n", max_len);
        return EXIT_FAILURE;
    }

    const size_t size = n*sizeof(*x);

    /* Allocate space for host copies of x, y */
    x = (double*)malloc(size); assert(x);
    y = (double*)malloc(size); assert(y);
    vec_init(x, y, n);

    printf("Computing the dot product of %d elements... ", n);
    result = dot(x, y, n);
    printf("result=%f\n", result);

    const double expected = ((double)n)/64;

    /* Check result */
    if ( fabs(result - expected) < 1e-5 ) {
        printf("Check OK\n");
    } else {
        printf("Check FAILED: got %f, expected %f\n", result, expected);
    }
    
    /* Cleanup */
    free(x); free(y);
    
    return EXIT_SUCCESS;
}
