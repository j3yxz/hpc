/* */
/****************************************************************************
 *
 * omp-loop.c - Restructure loops to remove dependencies
 *
 * Copyright (C) 2018, 2020 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 * Last modified on 2020-10-09 by Moreno Marzolla
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
 ****************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Three small functions used below; you should not need to know what
   these functions do */
int f(int a, int b, int c) { return (a+b+c)/3; }
int g(int a, int b) { return (a+b)/2; }
int h(int a) { return (a > 10 ? 2*a : a-1); }

/****************************************************************************/

/**
 * Shift the elements of array |a| of length |n| one position to the
 * right; the rightmost element of |a| becomes the leftmost element of
 * the shifted array.
 */
void vec_shift_right_seq(int *a, int n)
{
    int i;
    int tmp = a[n-1];
    for (i=n-1; i>0; i--) {
        a[i] = a[i-1];
    }
    a[0] = tmp;
}

void vec_shift_right_par(int *a, int n)
{
	
    /* [TODO] This function should be a parallel version of
       vec_shift_right_seq(). Note that it is not possible to remove
       the loop-carried dependency by aligning loop
       iterations. However, one solution is to use a temporary array
       b[] and split the loop into two loops: the first copies all
       elements of a[] in the shifted position of b[] (i.e., a[i] goes
       to b[i+1]; the rightmost element of a[] goes into b[0]). The
       second loop copies b[] into a[]. */
	int b[n];
#pragma omp parallel for
	for(int i=0; i<n; i++){
		b[i] = a[(i+1)%i];
	}

#pragma omp parallel for
	for(int i=0; i<n; i++){
		a[i] = b[i];
	}
}
/****************************************************************************/

void test1_seq(int *a, int *b, int *c, int n)
{
    int i;
    a[0] = h(0);
    b[0] = c[0] = 0;    
    b[1] = a[0] % 10;
    for (i=1; i<n-1; i++) {
        a[i] = h(i);
        b[i+1] = a[i] % 10;
        c[i] = a[i-1];
    }
    a[n-1] = h(n-1);
    c[n-1] = a[n-2];
}

void test1_par(int *a, int *b, int *c, int n)
{
    /* [TODO] This function should be a parallel version of
       test1_seq(). Loop-carried dependencies can be removed by
       aligning loop iterations. Pay attention to array elements that
       might fall outside the bounds of the aligned loop. */
}

/****************************************************************************/

/* This is a hack to convert 2D indexes to a linear index; this macro
   requires the existence of a variable "n" representing the number
   of columns of the matrix being indexed. A proper solution would
   be to use a C99-style cast:

   int (*AA)[n] = (int (*)[n])A;

   then, AA can be indexes as AA[i][j]. Unfortunately, this triggers a
   bug in gcc 5.4.0+openMP (works with gcc 8.2.0+OpenMP)
*/
#define IDX(i,j) ((i)*(n) + (j))

/* A is a nxn matrix */
void test2_seq(int *A, int n)
{
    int i, j;
    for (i=1; i<n; i++) {
        for (j=1; j<n-1; j++) {
            A[IDX(i,j)] = f(A[IDX(i-1,j-1)], A[IDX(i-1,j)], A[IDX(i-1,j+1)]);
        }
    }
}

void test2_par(int *A, int n)
{
    /* [TODO] This function should be a parallel vesion of
       test2_seq(). Suggestion: start by drawing the dependencies
       among the elements of matrix A[][] as they are computed. 
       Then, observe that one of the loops (which one?) can be
       parallelized as is with a #pragma opm parallel for directive. */
}

/****************************************************************************/

void test3_seq(int *A, int n)
{
    int i, j;
    for (i=1; i<n; i++) {
        for (j=1; j<n; j++) {
            A[IDX(i,j)] = g(A[IDX(i,j-1)], A[IDX(i-1,j-1)]);
        }
    }    
}

void test3_par(int *A, int n)
{
    /* [TODO] This function should be a parallel version of
       test3_seq(). Suggestion: start by drawing the dependencies
       among the elements of matrix A[][] as they are
       computed. Observe that it is not possible to put a "parallel
       for" directive on either loop.

       However, loops can be exchanged. If you do that... */
}

/****************************************************************************/

void test4_seq(int *A, int n)
{
    int i, j;
    for (i=1; i<n; i++) {
        for (j=1; j<n; j++) {
            A[IDX(i,j)] = f(A[IDX(i,j-1)], A[IDX(i-1,j-1)], A[IDX(i-1,j)]);
        }
    }    
}

void test4_par(int *A, int n)
{
    /* [TODO] This function should be a parallel version of
       test3_seq(). Suggestion: this is basically the same example
       shown on the slides, and can be solved by sweeping the matrix
       "diagonally" with the code fragment on the slides.

       There is a caveat: the code on the slides sweeps the _whole_
       matrix; in other words, variables i and j will assume all
       values starting from 0. The code of test4_seq() only process
       indexes where i>0 and j>0, so you need to add an "if" statement
       to skip the case where i==0 or j==0. */
}

void fill(int *a, int n)
{
    a[0] = 31;
    for (int i=1; i<n; i++) {
        a[i] = (a[i-1] * 33 + 1) % 65535;
    }
}

int array_equal(int *a, int *b, int n)
{
    for (int i=0; i<n; i++) {
        if (a[i] != b[i]) { return 0; }
    }
    return 1;
}

int main( void )
{
    const int N = 1024;
    int *a1, *b1, *c1, *a2, *b2, *c2;

    /* Allocate enough space for all tests */
    a1 = (int*)malloc(N*N*sizeof(int));
    b1 = (int*)malloc(N*sizeof(int));
    c1 = (int*)malloc(N*sizeof(int));

    a2 = (int*)malloc(N*N*sizeof(int));
    b2 = (int*)malloc(N*sizeof(int));
    c2 = (int*)malloc(N*sizeof(int));
    
    printf("vec_shift_right_par()\t"); fflush(stdout);
    fill(a1, N); vec_shift_right_seq(a1, N);
    printf("arrivo qua\n");
    fill(a2, N); vec_shift_right_par(a2, N);
    if ( array_equal(a1, a2, N) ) {
        printf("OK\n");
    } else {
        printf("FAILED\n");
    }

    /* test1 */
    printf("test1_par()\t\t"); fflush(stdout);
    test1_seq(a1, b1, c1, N);
    test1_par(a2, b2, c2, N);
    if (array_equal(a1, a2, N) &&
        array_equal(b1, b2, N) &&
        array_equal(c1, c2, N)) {
        printf("OK\n");
    } else {
        printf("FAILED\n");
    }

    /* test2 */
    printf("test2_par()\t\t"); fflush(stdout);
    fill(a1, N*N); test2_seq(a1, N);
    fill(a2, N*N); test2_par(a2, N);
    if (array_equal(a1, a2, N*N)) {
        printf("OK\n");
    } else {
        printf("FAILED\n");
    }
    
    /* test3 */
    printf("test3_par()\t\t"); fflush(stdout);
    fill(a1, N*N); test3_seq(a1, N);
    fill(a2, N*N); test3_par(a2, N);
    if (array_equal(a1, a2, N*N)) {
        printf("OK\n");
    } else {
        printf("FAILED\n");
    }

    /* test4 */
    printf("test4_par()\t\t"); fflush(stdout);
    fill(a1, N*N); test4_seq(a1, N);
    fill(a2, N*N); test4_par(a2, N);
    if (array_equal(a1, a2, N*N)) {
        printf("OK\n");
    } else {
        printf("FAILED\n");
    }

    free(a1); free(b1); free(c1); 
    free(a2); free(b2); free(c2);
    
    return EXIT_SUCCESS;
}
