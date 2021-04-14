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
#include <omp.h>
#include "hpc.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <x86intrin.h>
#include <float.h>

typedef int v8i __attribute__((vector_size(32)));
#define ViLEN (sizeof(v8i)/sizeof(int))
typedef float v8f __attribute__((vector_size(32)));
#define VfLEN (sizeof(v8f)/sizeof(float))
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
  //  const size_t mem_vlen_mul = (VfLEN - N%VfLEN)%VfLEN;
#if 1
    assert(!posix_memalign((void**)&P, __BIGGEST_ALIGNMENT__, D * ( N + omp_get_max_threads() ) * sizeof(*P) ) );
    assert(P);
#else
    P = (float*)malloc( D * N * sizeof(*P) );
#endif
    for (i=0; i<N; i++) {
        for (k=0; k<D; k++) {
            if (1 != scanf("%f", &(P[i*D + k]))) {
                fprintf(stderr, "FATAL: failed to get coordinate %d of point %d\n", k, i);
                exit(EXIT_FAILURE);
            }
        }
    }
    /*  se N non fosse multiplo di VLEN setto i valori eccessivi aggiunti uguali
        ai primi punti che si incontrano in modo da non interferire col risultato 
        dello skyline */
#if 0
    if(mem_vlen_mul) memcpy(P+N, P, D*mem_vlen_mul*sizeof(*P));
#endif
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
#if 1
int dominates_simd(float* P, int* v8i_s_p, int* index_ex, int* index_in, v8f* p_8 , v8f* q_8, int D )
{   
  //  printf("entro d_simd\n");
    int k, local_r=0;
    float tmp;

    v8i res;
    
    /*la funziona mi passa VLEN indici di _s e _q e devo vedere i punti associati se si dominano tra loro*/
    for(int i = 0; i<D ;i++){
     /*   int p0 = P[index_ex[0]*D+i];
        int p1 = P[index_ex[1]*D+i];
        int p2 = P[index_ex[2]*D+i];
        int p3 = P[index_ex[3]*D+i];
        int p4 = P[index_ex[4]*D+i];
        int p5 = P[index_ex[5]*D+i];
        int p6 = P[index_ex[6]*D+i];
        int p7 = P[index_ex[7]*D+i];
    */  *(p_8+i)=(v8f)_mm256_setr_ps(P[index_ex[0]*D+i], P[index_ex[1]*D+i], P[index_ex[2]*D+i], P[index_ex[3]*D+i], P[index_ex[4]*D+i], P[index_ex[5]*D+i], P[index_ex[6]*D+i], P[index_ex[7]*D+i]);
     //   v8f porcodio =(v8f)_mm256_setr_ps(p0, p1, p2, p3, p4, p5, p6, p7);
        *(q_8+i)=(v8f)_mm256_setr_ps(P[index_in[0]*D+i], P[index_in[1]*D+i], P[index_in[2]*D+i], P[index_in[3]*D+i], P[index_in[4]*D+i], P[index_in[5]*D+i], P[index_in[6]*D+i], P[index_in[7]*D+i]);
        //*(p_8 + i) = porcodio;
    }
    
 //   printf("P[ind_ex[0]]: [ %f ] __ P[ind_ex[7]]: [ %f ]\n\n",P[index_ex[0]*D], P[index_ex[7]*D]);
  //  printf("p_8 [%f] [%f] [%f] [%f] [%f] [%f] [%f] [%f]\n", (float)p_8[0][0], (float)(*p_8)[1], (float)(*p_8)[2], (float)(*p_8)[3], (float)(*p_8)[4], (float)(*p_8)[5], (float)(*p_8)[6], (float)(*p_8)[7]);
  //  printf("q_8 [%f] [%f] [%f] [%f] [%f] [%f] [%f] [%f]\n", (float)(*q_8)[0], (float)(*q_8)[1], (float)(*q_8)[2], (float)(*q_8)[3], (float)(*q_8)[4], (float)(*q_8)[5], (float)(*q_8)[6], (float)(*q_8)[7]);
   
    for(int shift=0; shift<ViLEN; shift++){
        /* The following loop could be merged, but the keep them separated
           for the sake of readability */
    //    printf("q_8 [%f] [%f] [%f] [%f] [%f] [%f] [%f] [%f]\n", (float)(*q_8)[0], (float)(*q_8)[1], (float)(*q_8)[2], (float)(*q_8)[3], (float)(*q_8)[4], (float)(*q_8)[5], (float)(*q_8)[6], (float)(*q_8)[7]);
        res=(v8i)_mm256_setzero_ps();
        for (k=0; k<D; k++) {
            res+= (p_8[k] < q_8[k]);
        }
     //   printf("res [%d] [%d] [%d] [%d] [%d] [%d] [%d] [%d] \n", res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7] );
        /* faccio in modo che se vi fosse anche una sola dimensione non dominata
           questa valga più del ciclo dopo */
        res*=D;

        for (k=0; k<D; k++) {
            res-= (p_8[k] > q_8[k]);
        }
        
     //   printf("res [%d] [%d] [%d] [%d] [%d] [%d] [%d] [%d] \n", res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7] );
        /* ora tolgo dal vettore v8i_s i punti che vengono dominati e aggiorno
           il contatore local_r */
     //   printf("shift [%d] \n", shift );
        for(k=0; k<ViLEN; k++){
            
            if(res[k] > 0 && v8i_s_p[index_in[ (k + shift) & 7]]){
    //            printf("index_in[ (k - shift) & 7] ::: [%d] \n", index_in[ (k - shift) & 7] );
                local_r++;
                v8i_s_p[index_in[(k + shift) & 7]] = 0;
            }
        }

        /* ora invece sposto tutti i punti di q_8 di una posizione indietro, dentro se stesso,
           in questo modo posso controllare tra di loro tutti i ViLEN punti passati a questa
           funzione contemporaneamente */
        for(k=0; k<D; k++){

            __m256i  v0 = _mm256_loadu_si256( (__m256i*)( ((float*)&(q_8[k]))+1 )   );
            tmp=q_8[k][0];
            _mm256_store_si256((__m256i*)&(q_8[k]), v0);
            q_8[k][7]=tmp; 
            
        }

    }

    //free(q_8);
    //free(p_8);

    return local_r;

}

#endif
int dominates_simd_v2(float* P, int* v8i_s_p, int* index_in, v8f* p_8 , v8f* q_8, int D )
{   
  //  printf("entro d_simd\n");
    int k, local_r=0;
    float tmp;

    v8i res;
    
    /*la funziona mi passa VLEN indici di _s e _q e devo vedere i punti associati se si dominano tra loro*/
    for(int i = 0; i<D ;i++){
     /*   int p0 = P[index_ex[0]*D+i];
        int p1 = P[index_ex[1]*D+i];
        int p2 = P[index_ex[2]*D+i];
        int p3 = P[index_ex[3]*D+i];
        int p4 = P[index_ex[4]*D+i];
        int p5 = P[index_ex[5]*D+i];
        int p6 = P[index_ex[6]*D+i];
        int p7 = P[index_ex[7]*D+i];
    */
     //   v8f porcodio =(v8f)_mm256_setr_ps(p0, p1, p2, p3, p4, p5, p6, p7);
        *(q_8+i)=(v8f)_mm256_setr_ps(P[index_in[0]*D+i], P[index_in[1]*D+i], P[index_in[2]*D+i], P[index_in[3]*D+i], P[index_in[4]*D+i], P[index_in[5]*D+i], P[index_in[6]*D+i], P[index_in[7]*D+i]);
        //*(p_8 + i) = porcodio;
    }
    
 //   printf("P[ind_ex[0]]: [ %f ] __ P[ind_ex[7]]: [ %f ]\n\n",P[index_ex[0]*D], P[index_ex[7]*D]);
  //  printf("p_8 [%f] [%f] [%f] [%f] [%f] [%f] [%f] [%f]\n", (float)p_8[0][0], (float)(*p_8)[1], (float)(*p_8)[2], (float)(*p_8)[3], (float)(*p_8)[4], (float)(*p_8)[5], (float)(*p_8)[6], (float)(*p_8)[7]);
  //  printf("q_8 [%f] [%f] [%f] [%f] [%f] [%f] [%f] [%f]\n", (float)(*q_8)[0], (float)(*q_8)[1], (float)(*q_8)[2], (float)(*q_8)[3], (float)(*q_8)[4], (float)(*q_8)[5], (float)(*q_8)[6], (float)(*q_8)[7]);
   
    for(int shift=0; shift<ViLEN; shift++){
        /* The following loop could be merged, but the keep them separated
           for the sake of readability */
    //    printf("q_8 [%f] [%f] [%f] [%f] [%f] [%f] [%f] [%f]\n", (float)(*q_8)[0], (float)(*q_8)[1], (float)(*q_8)[2], (float)(*q_8)[3], (float)(*q_8)[4], (float)(*q_8)[5], (float)(*q_8)[6], (float)(*q_8)[7]);
        res=(v8i)_mm256_setzero_ps();
        for (k=0; k<D; k++) {
            res+= (p_8[k] < q_8[k]);
        }
     //   printf("res [%d] [%d] [%d] [%d] [%d] [%d] [%d] [%d] \n", res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7] );
        /* faccio in modo che se vi fosse anche una sola dimensione non dominata
           questa valga più del ciclo dopo */
        res*=D;

        for (k=0; k<D; k++) {
            res-= (p_8[k] > q_8[k]);
        }
        
     //   printf("res [%d] [%d] [%d] [%d] [%d] [%d] [%d] [%d] \n", res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7] );
        /* ora tolgo dal vettore v8i_s i punti che vengono dominati e aggiorno
           il contatore local_r */
     //   printf("shift [%d] \n", shift );
        for(k=0; k<ViLEN; k++){
            
            if(res[k] > 0 && v8i_s_p[index_in[ (k + shift) & 7]]){
    //            printf("index_in[ (k - shift) & 7] ::: [%d] \n", index_in[ (k - shift) & 7] );
                local_r++;
                v8i_s_p[index_in[(k + shift) & 7]] = 0;
            }
        }

        /* ora invece sposto tutti i punti di q_8 di una posizione indietro, dentro se stesso,
           in questo modo posso controllare tra di loro tutti i ViLEN punti passati a questa
           funzione contemporaneamente */
        for(k=0; k<D; k++){

            __m256i  v0 = _mm256_loadu_si256( (__m256i*)( ((float*)&(q_8[k]))+1 )   );
            tmp=q_8[k][0];
            _mm256_store_si256((__m256i*)&(q_8[k]), v0);
            q_8[k][7]=tmp; 
            
        }

    }

    //free(q_8);
    //free(p_8);

    return local_r;

}
#if 1
int skyline_simd_for( const points_t *points, v8i* v8i_s )
{   

    const int n_threads = omp_get_max_threads();
    int thread_id;
    printf("n_threads %d\n",n_threads);
    const int D = points->D;
    const int N = points->N;
    float *P = points->P;

    int* v8i_s_p = (int*)v8i_s;
    int i = 0, r = N;
    int count_ex=0, count_in;
    int local_r[n_threads];



    //const v8i ones = (v8i)_mm256_set1_epi32(1);

    v8i index_ex;
    v8i index_in;
    /*[N/ViLEN +1]*/
  //  int index_in[N+1][ViLEN];
    
    v8f p_8[D];
    v8f q_8[D];

    
/*    const v8i ones = {1,1,1,1, 
                    1,1,1,1};
*/



    for (i=0; i<N/ViLEN +1; i++) {
        *(v8i_s+i) = (v8i)_mm256_set1_epi32(1); /* {1,1,1,1,
                    1,1,1,1}; */
    }

    /* mi sa che non serve */
    if(0 != N%ViLEN) *(v8i_s+i) =(v8i)_mm256_set1_epi32(1);

    *(v8i_s_p+N)=0;


/* qui non si può parallelizzare */

  //  printf("wot\n");

//    assert(! posix_memalign((void**)&p_8, __BIGGEST_ALIGNMENT__, D*sizeof(float)*VfLEN) );
 //   assert(! posix_memalign((void**)&q_8, __BIGGEST_ALIGNMENT__, D*sizeof(float)*VfLEN) );

#if __GNUC__ < 9
 #pragma omp parallel num_threads(n_threads) default(none) private(q_8, thread_id, count_in, index_in) shared(i, v8i_s_p, v8i_s, P, p_8, count_ex, index_ex, local_r) reduction(-:r)
#else
 #pragma omp parallel num_threads(n_threads) default(none) private(q_8, thread_id, count_in, index_in) shared(n_threads, D, N, P, i, v8i_s_p, v8i_s, p_8, count_ex, index_ex, local_r) reduction(-:r)
#endif
{    
    
    thread_id=omp_get_thread_num();
    count_in=0;
    for(int d=0; d<D; d++) P[(N + thread_id)*D+d] = -FLT_MAX;
    local_r[thread_id]=0;


   // assert(! posix_memalign((void**)&(p_8+thread_id), __BIGGEST_ALIGNMENT__, D*sizeof(float)*(VfLEN + 1)) );
  //  p_8=malloc(D*sizeof(float)*VfLEN);
  //  q_8=malloc(D*sizeof(float)*VfLEN);
    assert(! posix_memalign((void**)&(q_8), __BIGGEST_ALIGNMENT__, D*sizeof(float)*(VfLEN + 1)) );
   // printf("diogay %d\n", thread_id);


#pragma omp barrier
   
{
    do { 
    //    printf("i=[%d] \n",i);
#pragma omp master      
{
        count_ex=0;
        while(count_ex < ViLEN && i<N){
            if (v8i_s_p[i]) index_ex[count_ex++]=i;
            ++i;   
        }
        if (count_ex != ViLEN ) break;
        else {
            for(int d = 0; d<D ;d++){
                (p_8+i)=(v8f)_mm256_setr_ps(P[index_ex[0]*D+i], P[index_ex[1]*D+i], P[index_ex[2]*D+i], P[index_ex[3]*D+i], P[index_ex[4]*D+i], P[index_ex[5]*D+i], P[index_ex[6]*D+i], P[index_ex[7]*D+i]);
            }
        }
}

#pragma omp for firstprivate(p_8) schedule(static)
       for (int j=0; j<N; j++ ) {
            
            if (v8i_s_p[j]) {
                
                index_in[count_in++]=j;
                if(count_in == ViLEN) {
                    count_in=0;
               //     printf("for in j:[%d]\n", j);
                    /* faccio partire il task */
                    //#pragma omp task firstprivate(th_offset_in, th_offset_ex)

               //     printf(" ho fatto partire il task %d\n", thread_id );
                    local_r[thread_id]+=dominates_simd_v2(P, v8i_s_p, index_in, p_8, q_8, D);
             //      printf("__i:[%d] j:[%d] local_r:[%d]__\n",i,j,local_r[thread_id]); 
            //        printf("ho fatto spawnare sto cazzo di task? %d\n", thread_id);

                }       
            }

        }
#pragma omp for firstprivate(p_8) schedule(static)
        for(int j=0; j<n_threads; j++){
            if(count_in) {
            /* devo gestire i rimanenti*/
        //    printf("if count_in 323 %d\n", thread_id);
                while(count_in < ViLEN) index_in[count_in++]=N+thread_id;

                count_in=0;
                local_r[thread_id]+=dominates_simd_v2(P, v8i_s_p, index_in, p_8, q_8, D );

            }

    //    printf("th_offset_ex\n");
        }
#pragma omp barrier       
    } while(i<N);
 //   printf("338\n");
    /* stesso caso del for interno solo che aggiungo a mano nei punti mancanti un punto che non può influire 
       dentro a index_ex, ora devo fare una sola iterazione del for esterno */
    if(count_ex){ //printf("if count_ex 341\n");

        while(count_ex < ViLEN) index_ex[count_ex++]=N+thread_id;
        for(int d = 0; d<D ;d++){
                (p_8+i)=(v8f)_mm256_setr_ps(P[index_ex[0]*D+i], P[index_ex[1]*D+i], P[index_ex[2]*D+i], P[index_ex[3]*D+i], P[index_ex[4]*D+i], P[index_ex[5]*D+i], P[index_ex[6]*D+i], P[index_ex[7]*D+i]);
        }
     //   printf("344 dopo while count_ex:[%d]\n", count_ex);
        #pragma omp for firstprivate(p_8) schedule(static)
        for (int j=0; j<N; j++ ) {
            if (v8i_s_p[j]) {

                index_in[count_in++]=j;
                if(count_in == ViLEN) {
                    count_in=0;
                        /* faccio partire il task */
                    //#pragma omp task firstprivate(th_offset_in,th_offset_ex)
                    local_r[thread_id]+=dominates_simd_v2(P, v8i_s_p, index_in, p_8, q_8, D);
                }            
            }
        }
        
   //     printf("359 dopo for\n");
        #pragma omp for firstprivate(p_8) schedule(static)
        for(int j=0; j<n_threads; j++){
            if(count_in) {
            /* devo gestire i rimanenti*/
        //    printf("if count_in 323 %d\n", thread_id);
                while(count_in < ViLEN) index_in[count_in++]=N+thread_id;

                count_in=0;
                local_r[thread_id]+=dominates_simd_v2(P, v8i_s_p, index_in, p_8, q_8, D );

        }
        
    }
  //  printf("free?\n");

} /* fine di omp single */

    //free(p_8);
    free(q_8);
    r-=local_r[thread_id];
//#pragma omp barrier
}
    return r;
}


int skyline_simd( const points_t *points, v8i* v8i_s )
{   

    const int n_threads = omp_get_max_threads();
    int thread_id;
    printf("n_threads %d\n",n_threads);
    const int D = points->D;
    const int N = points->N;
    float *P = points->P;

    int* v8i_s_p = (int*)v8i_s;
    int i, r = N, th_offset_in=0,th_offset_ex=0;
    int count_ex=0, count_in=0;
    int local_r[n_threads];



    //const v8i ones = (v8i)_mm256_set1_epi32(1);

    int index_ex[N+1][ViLEN];/*[N/ViLEN +1]*/
    int index_in[N+1][ViLEN];
    
    v8f* p_8;
    v8f* q_8;

    
/*    const v8i ones = {1,1,1,1, 
                    1,1,1,1};
*/

    for(int d=0; d<D; d++) P[N*D+d] = -FLT_MAX;

    for (i=0; i<N/ViLEN +1; i++) {
        *(v8i_s+i) = (v8i)_mm256_set1_epi32(1); /* {1,1,1,1,
                    1,1,1,1}; */
    }

    /* mi sa che non serve */
    if(0 != N%ViLEN) *(v8i_s+i) =(v8i)_mm256_set1_epi32(1);

    *(v8i_s_p+N)=0;


/* qui non si può parallelizzare */

  //  printf("wot\n");

//    assert(! posix_memalign((void**)&p_8, __BIGGEST_ALIGNMENT__, D*sizeof(float)*VfLEN) );
 //   assert(! posix_memalign((void**)&q_8, __BIGGEST_ALIGNMENT__, D*sizeof(float)*VfLEN) );

#if __GNUC__ < 9
 #pragma omp parallel num_threads(n_threads) default(none) private(thread_id) shared(i, v8i_s_p, v8i_s, P, th_offset_in, th_offset_ex, p_8,q_8, count_in, count_ex, index_ex, index_in, local_r) reduction(-:r)
#else
 #pragma omp parallel num_threads(n_threads) default(none) private(p_8,q_8,thread_id) shared(n_threads, D, N, P, i, v8i_s_p, v8i_s, th_offset_in, th_offset_ex, count_in, count_ex, index_ex, index_in, local_r) reduction(-:r)
#endif
{    

    thread_id=omp_get_thread_num();

    local_r[thread_id]=0;


    assert(! posix_memalign((void**)&p_8, __BIGGEST_ALIGNMENT__, D*sizeof(float)*(VfLEN + 1)) );
  //  p_8=malloc(D*sizeof(float)*VfLEN);
  //  q_8=malloc(D*sizeof(float)*VfLEN);
    assert(! posix_memalign((void**)&q_8, __BIGGEST_ALIGNMENT__, D*sizeof(float)*(VfLEN + 1)) );
   // printf("diogay %d\n", thread_id);
#pragma omp barrier
#pragma omp master    
{
    for (i=0; i<N; i++) { 
    //    printf("i=[%d] \n",i);
        if (v8i_s_p[i]) { 

            index_ex[th_offset_ex][count_ex++]=i; //printf("if i=[%d] th_offset_ex:[%d]\n", i, th_offset_ex);
            if(count_ex == ViLEN){
                count_ex=0;

            /* qui si può paralellizzare */

            //    printf("for ex %d\n", thread_id);

//#pragma omp parallel num_threads(n_threads) default(none) shared(i, v8i_s, v8i_s_p, P, count_in, count_ex, index_th_offset, index_ex, index_in) reduction(+:local_r)

              // #pragma omp single
               for (int j=0; j<N; j++ ) {
                    
                    if (v8i_s_p[j]) {
                        
                        index_in[th_offset_in][count_in++]=j;
                        if(count_in == ViLEN) {
                            count_in=0;
                       //     printf("for in j:[%d]\n", j);
                            /* faccio partire il task */
                            //#pragma omp task firstprivate(th_offset_in, th_offset_ex)
{
                       //     printf(" ho fatto partire il task %d\n", thread_id );
                            local_r[thread_id]+=dominates_simd(P, v8i_s_p, index_ex[th_offset_ex], index_in[th_offset_in], p_8, q_8, D);
}                     //      printf("__i:[%d] j:[%d] local_r:[%d]__\n",i,j,local_r[thread_id]); 
                    //        printf("ho fatto spawnare sto cazzo di task? %d\n", thread_id);
                            th_offset_in++;
                        }       
                    }
#pragma omp taskwait                    /
                }
                
                if(count_in) {
                    /* devo gestire i rimanenti*/
                //    printf("if count_in 323 %d\n", thread_id);
                    while(count_in < ViLEN) index_in[th_offset_in][count_in++]=N;

                    count_in=0;
                    local_r[thread_id]+=dominates_simd(P, v8i_s_p, index_ex[th_offset_ex], index_in[th_offset_in], p_8, q_8, D );
                    th_offset_in++;
                }
                th_offset_ex++;
            //    printf("th_offset_ex\n");

            }
            
        }
        
    }
 //   printf("338\n");
    /* stesso caso del for interno solo che aggiungo a mano nei punti mancanti un punto che non può influire 
       dentro a index_ex, ora devo fare una sola iterazione del for esterno */
    if(count_ex){ //printf("if count_ex 341\n");

        while(count_ex < ViLEN) index_ex[th_offset_ex][count_ex++]=N;
     //   printf("344 dopo while count_ex:[%d]\n", count_ex);

        for (int j=0; j<N; j++ ) {
            if (v8i_s_p[j]) {

                index_in[th_offset_in][count_in++]=j;
                if(count_in == ViLEN) {
                    count_in=0;
                        /* faccio partire il task */
                    //#pragma omp task firstprivate(th_offset_in,th_offset_ex)
                    local_r[thread_id]+=dominates_simd(P, v8i_s_p, index_ex[th_offset_ex], index_in[th_offset_in], p_8, q_8, D);
                    th_offset_in++;
                }            
            }
        }
        
   //     printf("359 dopo for\n");
        if(count_in) {
                    /* devo gestire i rimanenti*/

            while(count_in < ViLEN) index_in[th_offset_in][count_in++]=N;

            /* count_in=0; ultima iterazione non devo aggiornare */
                local_r[thread_id]+=dominates_simd(P, v8i_s_p, index_ex[th_offset_ex], index_in[th_offset_in], p_8, q_8, D );
            /* index_th_offset++; posso anche non farla perché è l'ultima iterazione */
        }
        #pragma omp taskwait
    }
  //  printf("free?\n");

} /* fine di omp single */

    free(p_8);
    free(q_8);
    r-=local_r[thread_id];
//#pragma omp barrier
}
    return r;
}

#endif
/**
 * Compute the skyline of |points|. At the end, s[i] == 1 iff point
 * |i| belongs to the skyline. This function returns the number r of
 * points in to the skyline. The caller is responsible for allocating
 * a suitably sized array |s|.
 */
int skyline( const points_t *points, int *s )
{
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;
    int i, r = N, local_r = 0;

    for (i=0; i<N; i++) {
        s[i] = 1;
    }

    const int n_threads = omp_get_max_threads();
  //  printf("%d\n", n_threads);
/* qui non si può parallelizzare */
    for (i=0; i<N; i++) {

        if ( s[i] ) {

            /* qui si può paralellizzare */
#if __GNUC__ < 9
 #pragma omp parallel for num_threads(n_threads) default(none) shared(i, s, P) reduction(+:local_r)
#else
 #pragma omp parallel for num_threads(n_threads) default(none) shared(n_threads, D, N, P, i, s, r) reduction(+:local_r)
#endif
            for (int j=0; j<N; j++ ) {
                if ( s[j] && dominates( &(P[i*D]), &(P[j*D]), D ) ) {
                    s[j] = 0;
                    local_r++;
                }
            }
            r-=local_r;
            local_r=0;
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
    int* s;
    v8i* v8i_s;
    /* calcolo quanta memoria aggiuntiva devo allocare per far si che sia multiplo di VLEN */
    const size_t mem_vlen = points.N + (ViLEN - points.N%ViLEN)%ViLEN;
#if 1
    assert (!posix_memalign((void**)&s, __BIGGEST_ALIGNMENT__, mem_vlen * sizeof(int) ) );
    assert (!posix_memalign((void**)&v8i_s, __BIGGEST_ALIGNMENT__, (mem_vlen + omp_get_max_threads()) * sizeof(void*) ) );
#else
    s = (int*)malloc(points.N * sizeof(*s));
#endif
    assert(s);
    assert(v8i_s);
    const double tstart = hpc_gettime();
    const int r = skyline(&points, s);
    const double elapsed = hpc_gettime() - tstart;
    //printf("qui forse\n");

    //print_skyline(&points, s, r);

    fprintf(stderr,
            "\n\t%d points\n\t%d dimensione\n\t%d points in skyline\n\nExecution time %f seconds\n",
            points.N, points.D, r, elapsed);

    const double ts = hpc_gettime();
    const int r_simd = skyline_simd(&points, v8i_s);
    const double el = hpc_gettime() - ts;
    fprintf(stderr,
        "\n\t%d points\n\t%d dimensione\n\t%d points in skyline\n\nExecution time %f seconds\n",
        points.N, points.D, r_simd, el);
    free_points(&points);
    free(s);
    free((int*)v8i_s);
    return EXIT_SUCCESS;
}



















#if 0


typedef struct Next_index {
        int index;
        struct Next_index* next;
    } next_index;

int sk_simd( const points_t *points, int *s )
{

    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;
    int i, r = N, local_r = 0;
    v8i v8i_s[(N+VLEN-1)/VLEN], v8i_t_inner, v8i_t_extern;



/* questa lista mi serve per tener traccia di quale sarà il prossimo punto da analizzare */

    next_index* s_index_list_head=malloc(sizeof(next_index));
    next_index* s_index_list_tmp=s_index_list_head;

    for(i=0; i<N; i++){
        s_index_list_tmp->index=i;
        s_index_list_tmp->next=malloc(sizeof(next_index));
        s_index_list_tmp=s_index_list_tmp->next;
        s_index_list_tmp->next=NULL;
    }
    s_index_list_tmp=s_index_list_head;
    for (i=0; i<N; i+=VLEN) {
        v8i_s[i] = _mm256_set_epi32(1);
    }
    next_index* s_index_list_tmp=s_index_list_head;
    const int n_threads = omp_get_max_threads();



    /* qui non si può parallelizzare */
    do {
        v8i_t_extern=_mm256_set1_epi32(0,1,2,3,4,5,6,7)
        

    } while (s_index_list_tmp->next) {
            /* qui si può paralellizzare --forse--*/
        for(i=0; i<VLEN; i++) s_index_list_tmp=s_index_list_tmp->next;
/*
#if __GNUC__ < 9
 #pragma omp parallel for num_threads(n_threads) default(none) shared(i, s, P) reduction(+:local_r)
#else
 #pragma omp parallel for num_threads(n_threads) default(none) shared(n_threads, D, N, P, i, s, r) reduction(+:local_r)
#endif
*/
        next_index* s_index_list_inwhile=s_index_list_head;
        while (s_index_list_inwhile) {
            next_index* tmp = s_index_list_inwhile;
            for(i=0; i<VLEN; i++) {
                tmp=tmp->next;
                if(tmp == NULL) break;
            }
            if(i<VLEN){
                /* significa che è l'ultima iterazione del while interno da fare */
                /* qua posso mettere una barriera tra task in modo che l'ultimo aggiorni il tutto */
            } else {
                /* significa che ci sono altre iterazioni da fare */
                v8i_t_inner= _mm256_set_epi32(s_index_list_inwhile->index,
                                            s_index_list_inwhile->next->index,
                                            s_index_list_inwhile->next->next->index,
                                            s_index_list_inwhile->next->next->next->index,
                                            s_index_list_inwhile->next->next->next->next->index,
                                            s_index_list_inwhile->next->next->next->next->next->index,
                                            s_index_list_inwhile->next->next->next->next->next->next->index,
                                            s_index_list_inwhile->next->next->next->next->next->next->next->index )
       
                v8i v8i_d_res = dominates_simd(P, v8i_t_extern, v8i_t_inner, D )
                
                local_r=update_next_index_list( v8i_d_res ,&s_index_list_head, prev);
                
                
            
            }
            if(i == 0) {
                /* significa che non è risultato i<VLEN e bisogna aggiornare i risultati del while */
            }
            prev=s_index_list_inwhile;
            s_index_list_inwhile=tmp->next;
            
            
        }
        
        r-=local_r;



        if(s_index_list_tmp == NULL) {
            /* significa che il punto */
        }
    }

    return r;
}

int update_next_index_list( v8i v8i_d_res, next_index** s_index_list_head, next_index* prev){

    int local_r=0;
    next_index* tmp;

    if(prev != NULL ){
        /* questo if-else più esterni fanno sostanzialmente la stessa cosa ma uno lo fa nel caso
            non ci sia bisogno di aggiornare la testa, mentre l'altro si occupa anche del caso in
            cui bisogna eliminare il primo elemento della lista 
        */
        for(int i=0; i<VLEN; i++){
            if(v8i_d_res[i]>-1) {
                while(prev->next->index != v8i_d_res[i]) prev=prev->next;
                tmp=prev->next-next;
                free(prev->next)
                prev->next=tmp;
                local_r++;
            }
        }
    } else {
        /* qui significa che prev era NULL e dunque devo aggiornare s_index_list_head */
        for(int i=0; i<VLEN; i++){
            if(v8i_d_res[i]==(*s_index_list_head)->index){
                /* significa che il primo elemento della lista è da eliminare */
                prev=*s_index_list_head;
                *s_index_list_head=(*s_index_list_head)->next;
                free(prev);
                prev=NULL;
                local_r++;
            } else {
                /* l'if successivo serve per impostare prev una sola volta */
                if(prev==NULL) prev=*s_index_list_head;

                if(v8i_d_res[i]>-1) {
                    while(prev->next->index != v8i_d_res[i]) prev=prev->next;
                    tmp=prev->next-next;
                    free(prev->next)
                    prev->next=tmp;
                    local_r++;
                }
            }
        }
    }

    return local_r;
}





#endif