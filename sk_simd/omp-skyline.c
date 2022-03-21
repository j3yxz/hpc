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
#if 0
int dominates_simd(float* P, int* v8i_s_p, int* index_ex, int* index_in, v8f* p_8 , v8f* q_8, int D )
{   

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
        *(q_8+i)=(v8f)_mm256_setr_ps(P[index_in[0]*D+i], P[index_in[1]*D+i], P[index_in[2]*D+i], P[index_in[3]*D+i], P[index_in[4]*D+i], P[index_in[5]*D+i], P[index_in[6]*D+i], P[index_in[7]*D+i]);

    }
       
    for(int shift=0; shift<ViLEN; shift++){
        /* The following loop could be merged, but the keep them separated
           for the sake of readability */
        res=(v8i)_mm256_setzero_ps();
        for (k=0; k<D; k++) {
            res+= (p_8[k] < q_8[k]);
        }
        /* faccio in modo che se vi fosse anche una sola dimensione non dominata
           questa valga più del ciclo dopo */
        res*=D;

        for (k=0; k<D; k++) {
            res-= (p_8[k] > q_8[k]);
        }
        
        /* ora tolgo dal vettore v8i_s i punti che vengono dominati e aggiorno
           il contatore local_r */
        for(k=0; k<ViLEN; k++){
            
            if(res[k] > 0 && v8i_s_p[index_in[ (k + shift) & 7]]){
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

    return local_r;

}

#endif
int dominates_simd_v2(float* P, int* v8i_s_p, int* index_in, v8f* p_8 , v8f* q_8, int D )
{   

    int k, local_r=0;
    float tmp;

    v8i res;
    
    /* la funziona mi passa VLEN indici (p_8 e q_8) e devo vedere i punti associati se si dominano tra loro */
    for(int i = 0; i<D ;i++){
        *(q_8+i)=(v8f)_mm256_setr_ps(P[index_in[0]*D+i], P[index_in[1]*D+i], P[index_in[2]*D+i], P[index_in[3]*D+i], P[index_in[4]*D+i], P[index_in[5]*D+i], P[index_in[6]*D+i], P[index_in[7]*D+i]);

    }
    for(int shift=0; shift<ViLEN; shift++){
        res=(v8i)_mm256_setzero_ps();
        for (k=0; k<D; k++) {
            res+= (p_8[k] < q_8[k]);
        }
        /* faccio in modo che se vi fosse anche una sola dimensione non dominata
           questa valga più del ciclo dopo */
        res*=D;

        for (k=0; k<D; k++) {
            res-= (p_8[k] > q_8[k]);
        }
        
        /* ora tolgo dal vettore v8i_s i punti che vengono dominati e aggiorno
           il contatore local_r */
        for(k=0; k<ViLEN; k++){
            
            if(res[k] > 0 && v8i_s_p[index_in[ (k + shift) & 7]]){
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

    return local_r;

}

int skyline_simd_for( const points_t *points, v8i* v8i_s )
{   

    const int n_threads = omp_get_max_threads();
    int thread_id;

    const int D = points->D;
    const int N = points->N;
    const int local_N=N;
    float *P = points->P;

    int* v8i_s_p = (int*)v8i_s;
    int i = 0, r = N;
    int count_ex=0, count_in;
    int local_r[n_threads];
    int index_ex[ViLEN]; 
    v8f p_8[D];
    v8f* q_8;





    for (i=0; i<N/ViLEN +1; i++) {
        *(v8i_s+i) = (v8i)_mm256_set1_epi32(1); /* {1,1,1,1,1,1,1,1}; */
    }

    if(0 != N%ViLEN) *(v8i_s+i) =(v8i)_mm256_set1_epi32(1);
    /* tolgo i punti aggiunti "in più" */
    for(i=0; i<n_threads; i++) *(v8i_s_p+N+i)=0;

    i=0;

#if __GNUC__ < 9
 #pragma omp parallel num_threads(n_threads) default(none) private(q_8, thread_id, count_in) firstprivate(local_N) shared(i, v8i_s_p, v8i_s, P, p_8, count_ex, index_ex, local_r) reduction(-:r)
#else
 #pragma omp parallel num_threads(n_threads) default(none) private(q_8, thread_id, count_in) firstprivate(local_N) shared(n_threads, D, N, P, i, v8i_s_p, v8i_s, p_8, count_ex, index_ex, local_r) reduction(-:r)
#endif
{    
    
    thread_id=omp_get_thread_num();
    int index_in[ViLEN];
    int local_i;
    count_in=0;
    /* inizializzo i punti "in più" a valori che non possano alterare il risultato finale */
    for(int d=0; d<D; d++) P[(N + thread_id)*D+d] = -FLT_MAX;
    local_r[thread_id]=0;
	
    /* visto che nella funzione dominates_simd_v2 che verrà usata si copiano VfLEN elementi a partire dall'indice 1, per sicurezza q_8 lo alloco lungo 9 */
    assert(! posix_memalign((void**)&(q_8), __BIGGEST_ALIGNMENT__, D*sizeof(float)*(VfLEN + 1)) );



#pragma omp barrier

    do { 

#pragma omp master
{
        /* questo è il ciclo in cui il master riempie p_8 di punti da passare alla funzione dominates_simd_v2
        servono ViLEN (8) punti affinché si possa procedere al passaggio successivo*/
        count_ex=0;
        while(count_ex < ViLEN && i<N){

            if (v8i_s_p[i]){ 

            	index_ex[count_ex++]=i;

            }
            ++i;   
        }
        /* controllo aggiuntivo nel caso in cui si sia uscito dal while precedente a causa di "i<N", in generale viene eseguita la parte "else" */
        if (count_ex != ViLEN ){

        	while(count_ex < ViLEN) index_ex[count_ex++]=N+thread_id;
	        for(int d = 0; d<D ;d++){
	                *(p_8+d)=(v8f)_mm256_setr_ps(P[index_ex[0]*D+d], P[index_ex[1]*D+d], P[index_ex[2]*D+d], P[index_ex[3]*D+d], P[index_ex[4]*D+d], P[index_ex[5]*D+d], P[index_ex[6]*D+d], P[index_ex[7]*D+d]);
	        }
			
        } else {
            for(int d = 0; d<D ;d++){
                *(p_8+d)=(v8f)_mm256_setr_ps(P[index_ex[0]*D+d], P[index_ex[1]*D+d], P[index_ex[2]*D+d], P[index_ex[3]*D+d], P[index_ex[4]*D+d], P[index_ex[5]*D+d], P[index_ex[6]*D+d], P[index_ex[7]*D+d]);
            }
        }
}

#pragma omp barrier
/* in questo ciclo for ogni thread cerca 8 punti da inserire in q_8 per poterli paragonare con p_8, lo fa utilizzando la variabile "index_in" */
#pragma omp for schedule(static)
        for (int j=0; j<N; j++ ) {
            if (v8i_s_p[j]) {
                
                index_in[count_in++]=j;
                if(count_in == ViLEN) {
                    count_in=0;

                    local_r[thread_id]+=dominates_simd_v2(P, v8i_s_p, index_in, p_8, q_8, D);
                }       
            }
        }
/* nel caso in cui il for precedente sia finito con una iterazione "mancata" ovvero la condizione j<N è diventata falsa ma vi erano dei punti
    pronti per essere passati alla "dominates_simd_v2" (ovvero count_in era rimasta maggiore di 0), ogni thread fa una iterazione in più del ciclo
    for precedente */
#pragma omp for schedule(static)
        for(int j=0; j<n_threads; j++){
            if(count_in) {
                /* qua vengono inseriti i punti "fasulli" che hanno tutte le coordinate pari a -MAX_FLT in q_8 */
                while(count_in < ViLEN) index_in[count_in++]=N+thread_id;

                count_in=0;

                local_r[thread_id]+=dominates_simd_v2(P, v8i_s_p, index_in, p_8, q_8, D );
            }
        }
/* modo piuttosto brutto per uscire dal do while più esterno.. in generale inizialmente per evitare sta cosa ho usato i task e la cosa andava bene
finché i dati rimasti in cache su un processore non venivano usati sull'altro processore visto che non ho saputo avere più controllo nei task (i tempi
di esecuzione diventavano assurdi in certi casi).
Ho preferito fare questa versione che potessere sfruttare tutti i 12 thread senza "ripercussioni" ma anche nella versione coi task se si avviava il 
programma con numactl --cpunodebind=0 (oppure 1) i tempi di esecuzione erano molto simili (ovviamente considerando che c'erano solo 6 core a quel punto).
In realtà però anche in questa versione il problema di avere un ambiente numa si fa sentire, anche se in maniera più contenuta. (Una possibile soluzione
è quella di copiare l'insieme di tutti i punti per ogni processore o per ogni core) */
#pragma omp atomic read 
        local_i=i;
#pragma omp barrier
    } while(local_i<local_N);


    free(q_8);
    r-=local_r[thread_id];
/* fine regione parallela */

}
    return r;
}

#if 0
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

    for(i=0; i<n_threads; i++) *(v8i_s_p+N+i)=0;



#if __GNUC__ < 9
 #pragma omp parallel num_threads(n_threads) default(none) private(thread_id) shared(i, v8i_s_p, v8i_s, P, th_offset_in, th_offset_ex, p_8,q_8, count_in, count_ex, index_ex, index_in, local_r) reduction(-:r)
#else
 #pragma omp parallel num_threads(n_threads) default(none) private(p_8,q_8,thread_id) shared(n_threads, D, N, P, i, v8i_s_p, v8i_s, th_offset_in, th_offset_ex, count_in, count_ex, index_ex, index_in, local_r) reduction(-:r)
#endif
{    

    thread_id=omp_get_thread_num();

    local_r[thread_id]=0;


    assert(! posix_memalign((void**)&p_8, __BIGGEST_ALIGNMENT__, D*sizeof(float)*(VfLEN + 1)) );

    assert(! posix_memalign((void**)&q_8, __BIGGEST_ALIGNMENT__, D*sizeof(float)*(VfLEN + 1)) );

#pragma omp barrier
#pragma omp master    
{
    for (i=0; i<N; i++) { 

        if (v8i_s_p[i]) { 

            index_ex[th_offset_ex][count_ex++]=i; /
            if(count_ex == ViLEN){
                count_ex=0;

               for (int j=0; j<N; j++ ) {
                    
                    if (v8i_s_p[j]) {
                        
                        index_in[th_offset_in][count_in++]=j;
                        if(count_in == ViLEN) {
                            count_in=0;

                            local_r[thread_id]+=dominates_simd(P, v8i_s_p, index_ex[th_offset_ex], index_in[th_offset_in], p_8, q_8, D);

                            th_offset_in++;
                        }       
                    }
#pragma omp taskwait
                }
                
                if(count_in) {

                    while(count_in < ViLEN) index_in[th_offset_in][count_in++]=N;

                    count_in=0;
                    local_r[thread_id]+=dominates_simd(P, v8i_s_p, index_ex[th_offset_ex], index_in[th_offset_in], p_8, q_8, D );
                    th_offset_in++;
                }
                th_offset_ex++;
            }
            
        }
        
    }

    /* stesso caso del for interno solo che aggiungo a mano nei punti mancanti un punto che non può influire 
       dentro a index_ex, ora devo fare una sola iterazione del for esterno */
    if(count_ex){ //printf("if count_ex 341\n");

        while(count_ex < ViLEN) index_ex[th_offset_ex][count_ex++]=N;


        for (int j=0; j<N; j++ ) {
            if (v8i_s_p[j]) {

                index_in[th_offset_in][count_in++]=j;
                if(count_in == ViLEN) {
                    count_in=0;
                        /* faccio partire il task */
                    local_r[thread_id]+=dominates_simd(P, v8i_s_p, index_ex[th_offset_ex], index_in[th_offset_in], p_8, q_8, D);
                    th_offset_in++;
                }            
            }
        }

        if(count_in) {
                    /* devo gestire i rimanenti*/

            while(count_in < ViLEN) index_in[th_offset_in][count_in++]=N;

            /* count_in=0; ultima iterazione non devo aggiornare */
                local_r[thread_id]+=dominates_simd(P, v8i_s_p, index_ex[th_offset_ex], index_in[th_offset_in], p_8, q_8, D );
            /* index_th_offset++; posso anche non farla perché è l'ultima iterazione */
        }
        #pragma omp taskwait
    }


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
    const size_t mem_vlen = points.N + omp_get_max_threads() + ( points.N + omp_get_max_threads() )%ViLEN;
#if 1
    assert (!posix_memalign((void**)&s, __BIGGEST_ALIGNMENT__, points.N * sizeof(int) ) );
    assert (!posix_memalign((void**)&v8i_s, __BIGGEST_ALIGNMENT__, (mem_vlen) * sizeof(void*) ) );
#else
    s = (int*)malloc(points.N * sizeof(*s));
#endif
    assert(s);
    assert(v8i_s);

    const double tstart = hpc_gettime();
    const int r = skyline(&points, s);
    const double elapsed = hpc_gettime() - tstart;


    //print_skyline(&points, s, r);

    fprintf(stderr,
            "\n(omp)\n\t%d points\n\t%d dimensione\n\t%d points in skyline\n\nExecution time %f seconds\n",
            points.N, points.D, r, elapsed);
    if(points.D < 20){
        const double ts = hpc_gettime();
        const int r_simd = skyline_simd_for(&points, v8i_s);
        const double el = hpc_gettime() - ts;
        fprintf(stderr,
            "\n(simd)\n\t%d points\n\t%d dimensione\n\t%d points in skyline\n\nExecution time %f seconds\n",
        points.N, points.D, r_simd, el);
    } else {
        fprintf(stderr, "Versione SIMD non utilizzata\n");
    }
#if 0
    const double t = hpc_gettime();
    const int r_s = skyline_simd(&points, v8i_s);
    const double e = hpc_gettime() - t;
    fprintf(stderr,
        "\n\t%d points\n\t%d dimensione\n\t%d points in skyline\n\nExecution time %f seconds\n",
        points.N, points.D, r_s, e);
#endif

    free_points(&points);
    free(s);
    free((int*)v8i_s);
    return EXIT_SUCCESS;
}
