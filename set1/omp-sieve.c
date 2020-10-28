/* */
/****************************************************************************
 *
 * omp-sieve.c - Parallel implementation of the Sieve of Eratosthenes
 *
 * Copyright (C) 2018 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 * --------------------------------------------------------------------------
 *
 * This program counts the prime numbers in the set {2, ..., n} using
 * the sieve of Eratosthenes.
 *
 * Compile with:
 *
 * gcc -std=c99 -Wall -Wpedantic -fopenmp omp-sieve.c -o omp-sieve
 *
 * Run with:
 *
 * ./omp-sieve [n]
 *
 * You should expect the following results:
 *
 *                     num. of primes
 *             n     in {2, ... n}
 *   -----------    --------------
 *             1                 0
 *            10                 4
 *           100                25
 *          1000               168
 *         10000              1229
 *        100000              9592
 *       1000000             78498
 *      10000000            664579
 *     100000000           5761455
 *    1000000000          50847534
 *   10000000000                 ?  <- don't try this on the server!
 *                                     it allocates ~10GB of RAM
 *
 ****************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h> /* for memset() */

/* Mark all mutliples of |p| in the set {|from|, ..., |to|-1}; return how
   many numbers have been marked for the first time. |from| does not
   need to be a multiple of |p|. */
long mark( char *isprime, long from, long to, long p )
{
    long nmarked=0;
    
    from = ((from + p - 1)/p)*p; /* start from the lowest multiple of p that is >= from */
/******************************/
    //const long n = 1000000l;
    const int n_threads = omp_get_max_threads();
#if __GNUC__ < 9
 #pragma omp parallel for num_threads(n_threads) default(none) shared(from,to,isprime,p) reduction(+:nmarked) 
#else
 #pragma omp parallel for num_threads(n_threads) default(none) shared(n_threads,from,to,isprime,p) reduction(+:nmarked)
#endif
    for ( long x=from; x<to; x+=p ) {
        if (isprime[x] == 1) {
            isprime[x] = 0;
  //          printf("(%d):__%ld__\n ehm %ld",omp_get_thread_num(),x,nmarked[omp_get_thread_num()]);
            nmarked++;
        }
    }
//qua il pool di thread ha già finito e sta operando un thread solo

    return nmarked;
}

int main( int argc, char *argv[] )
{
    long n = 1000000l, nprimes, i;
    char *isprime;
    
    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc == 2 ) {
        n = atol(argv[1]);
    }

    if (n > (1ul << 31)) {
        fprintf(stderr, "FATAL: n too large\n");
        return EXIT_FAILURE;
    }
    
    isprime = (char*)malloc(n+1); assert(isprime);
    /* Initialize isprime[] to 1 */
    memset(isprime, 1, n+1);
    nprimes = n-1;
    const double tstart = omp_get_wtime();
    /* main iteration of the sieve */
    for (i=2; i*i <= n; i++) {
        if (isprime[i]) {
            nprimes -= mark(isprime, i*i, n+1, i);
        }
    }
    const double elapsed = omp_get_wtime() - tstart;
    /* Uncomment to print the list of primes */
   /*
    for (i=2; i<=n; i++) {
        if (isprime[i]) {printf("%ld ", i);}
    }
    printf("\n");
    */
    free(isprime);
    printf("There are %ld primes in {2, ..., %ld}\n", nprimes, n);
    printf("Elapsed time: %f\n", elapsed);
    return EXIT_SUCCESS;
}
