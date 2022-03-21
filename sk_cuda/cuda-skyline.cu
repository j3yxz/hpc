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
#define BLKDIM_1 32
#define MAX_D 1024 /* serve perché non posso creare array dinamici all'interno dei kernel */
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

/* in questo kernel accedo in memoria globale nel modo più banale possibile, prendendo D thread che copiano ognuno una coordinata del punto i BLKDIM volte in memoria shered
    dalla quale poi ogni thread si copia in locale ogni coordinata del punto i senza race condition */
__global__ void kernel_0(int i, int N, int D, int *d_s, float *d_P){

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float p[];
    int k;
    float temp;
    float local_p[MAX_D];
    /* per prima cosa metto in local_p il tutte le dimensioni del punto i-esimo che deve cercare di dominare il punto che ogni thread esamina in base al suo tid */
    
    /* metto il punti i in memoria shared da cui ogni thread lo preleva in modo parallelo mettendolo in locale */
    if(threadIdx.x < D){
        /* per accedere in memoria globale una sola volta anziché BLKDIM (credo che dovrebbe ottimizzarlo il compilatore, ma per sicurezza lo lascio) */
        temp=d_P[i*D + threadIdx.x];
        for(k=0; k<BLKDIM_1; k++) {
            p[threadIdx.x * blockDim.x + k] = temp;

        }

    }
__syncthreads();
    for(k=0; k<D; k++){
        local_p[k] = p[threadIdx.x + k*blockDim.x];
    }

    /* a questo punto controllo effettivamente che il punto i-esimo domini il punto d_P[tid] */
    if( tid >= N || tid == i || d_s[tid] == 0){
        return;
    } else {
        //dominates d_s[p] & d_s[tid]

        for (k=0; k<D; k++) {
            if (local_p[k] < d_P[tid*D+k]) {
                return;
            }
        }
        for (k=0; k<D; k++) {
            if (local_p[k] > d_P[tid*D+k]) {

                d_s[tid]=0;
                return;
            }
        }
    }
}

int skyline_cuda_0( const points_t *points, int *s )
{
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;
    /* visto che la memoria shared è limitata e dipende da D quanta me ne serve preferisco usare una dimensione del blocco più piccolo
        (con BLKDIM_1 = 1024 e D = 10 non c'è abbastanza memoria shared per eseguire il kernel */
    const int nblocks=1+(N-1)/BLKDIM_1;

    int i, r = 0;


    for (i=0; i<N; i++) {
        s[i] = 1;
    }
    int *d_s;
    float *d_P;

    cudaSafeCall( cudaMalloc( (void**)&d_s, N*sizeof(int)) );
    cudaSafeCall( cudaMalloc( (void**)&d_P, N*D*sizeof(float)) );

    cudaSafeCall( cudaMemcpy( d_s, s, N*sizeof(int), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy( d_P, P, N*D*sizeof(float), cudaMemcpyHostToDevice) );

    
    for (i=0; i<N; i++) {
        if ( s[i] ) {
            kernel_0<<<nblocks,BLKDIM_1,D*BLKDIM_1*sizeof(float)>>>(i, N, D, d_s, d_P);

            cudaSafeCall( cudaMemcpy( s, d_s, N*sizeof(int), cudaMemcpyDeviceToHost) );
            cudaSafeCall( cudaMemcpy( d_s, s, N*sizeof(int), cudaMemcpyHostToDevice) );
        }
    }
    
    for(i=0;i<N;i++){
        if(s[i]) r++;
    }

    cudaFree(d_s);
    cudaFree(d_P);

    return r;
}
/* questa versione è sostanzialmente identica alla versione 0, l'unica differenza è che "ruoto" l'insieme dei punti in modo che i thread possano accedere 
    in memoria globale in maniera contigua e teoricamente così migliorare le prestazioni a costo di dover riorganizzare l'intero input di dati 
    le uniche differenze nel kernel sono in (1), (2) e (3) quando accedo al vettore globale, nella funzione dell'host l'unica aggiunta è un for per riorganizzare i punti */
__global__ void kernel_1(int i, int N, int D, int *d_s, float *d_P){

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float p[];
    int k;
    float temp;
    float local_p[MAX_D];

    if(threadIdx.x < D){
        /* (1) */
        temp=d_P[i+N*threadIdx.x];
        for(k=0; k<BLKDIM_1; k++) {
            p[threadIdx.x * blockDim.x + k] = temp;
        }
    }
__syncthreads();
    for(k=0; k<D; k++){
        local_p[k] = p[threadIdx.x + k*blockDim.x];
    }

    if( tid >= N || tid == i || d_s[tid] == 0){
        return;
    } else {
        //dominates d_s[p] & d_s[tid]

        for (k=0; k<D; k++) {
            /* (2) */
            if (local_p[k] < d_P[tid+(N*k)]) {
                return;
            }
        }
        for (k=0; k<D; k++) {
            /* (3) */
            if (local_p[k] > d_P[tid+(N*k)]) {
                d_s[tid]=0;
                return;
            }
        }
    }
}



int skyline_cuda_1( const points_t *points, int *s )
{
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;
    const int nblocks=1+(N-1)/BLKDIM_1;
    int i,k,r = 0;

    float *h_i;
    h_i=(float*)calloc(N*D,sizeof(float));
    


    for (i=0; i<N; i++) {
        s[i] = 1;
    }
    /* qua vengono riorganizzati i punti, verrà usato questo vettore anziché P */
    for(k=0; k<D; k++){
        for(i=0; i<N; i++){
            h_i[i+N*k] = P[k+i*D];
        }
    }
    int *d_s;
    float *d_i;

    cudaSafeCall( cudaMalloc( (void**)&d_s, N*sizeof(int)) );
    cudaSafeCall( cudaMalloc( (void**)&d_i, N*D*sizeof(float)) );

    cudaSafeCall( cudaMemcpy( d_s, s, N*sizeof(int), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy( d_i, h_i, N*D*sizeof(float), cudaMemcpyHostToDevice) );

    
    for (i=0; i<N; i++) {
        if ( s[i] ) {

            kernel_1<<<nblocks,BLKDIM_1,D*BLKDIM_1*sizeof(float)>>>(i, N, D, d_s, d_i);

            cudaSafeCall( cudaMemcpy( s, d_s, N*sizeof(int), cudaMemcpyDeviceToHost) );
            cudaSafeCall( cudaMemcpy( d_s, s, N*sizeof(int), cudaMemcpyHostToDevice) );
        }
    }

    
    for(i=0;i<N;i++){
        if(s[i]) r++;
    }
    cudaFree(d_s);
    cudaFree(d_i);
    free(h_i);
    return r;
}

/* invece nelle seguenti 2 versioni quello che ho cercato di fare è accedere meno volte possibile, almeno per il punto i, alla memoria globale e 
farlo in maniera contigua, la cosa positiva è che mi serve una quantità minore di memoria shared (1.5*BLKDIM) ma quella negativa sono tante sincronizzazioni 
nel processo, inizialmente speravo che con blocchi da 32 thread non dovessi sincronizzare visto che è un singolo warp che accede in memoria contigua ma 
anche in quel caso servono le sincronizzazioni, dunque ho optato a quel punto a fare blocchi di thread il più grandi possibili in modo da fare meno accessi 
in memoria globale visto che raddopiando il numero di thread per blocco devo fare solo 1 accesso in più */
__global__ void kernel_2(int i, int N, int D, int *d_s, float *d_P){

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float p[];
    int k,f;
    float glob_temp;
    float temp;
    float local_p[MAX_D];
    
/* accedo in memoria globale solo 1 volta, poi sfrutto la memoria shared per costruirmi 1 vettore locale,
 che avrà valori pari alla dimensione i-esima di P */
    glob_temp=d_P[i*D + threadIdx.x];
//__syncthreads();
    for(f=0; f<D; f++){
        
        /* ora ogni volta ricopio in memoria shared il vettore glob_temp (quello preso dalla memoria globale) spostandosi di posizione (da 0 a +1 +2 +4 +8 +16..)
           ricopiando temporaneamente il vettore ottenuto ogni volta in temp e procedendo da temp */
__syncthreads(); 
        temp=glob_temp;
        p[threadIdx.x] = temp;
__syncthreads();
        glob_temp = p[threadIdx.x + 1];
__syncthreads();
        p[threadIdx.x + 1] = temp;
__syncthreads();
        temp=p[threadIdx.x];
__syncthreads();
        p[threadIdx.x + 2] = temp;
__syncthreads();
        temp=p[threadIdx.x];
__syncthreads();
        p[threadIdx.x + 4] = temp;
__syncthreads();
        temp=p[threadIdx.x];
__syncthreads();
        p[threadIdx.x + 8] = temp;
__syncthreads();
        temp=p[threadIdx.x];
__syncthreads();
        p[threadIdx.x + 16] = temp;
__syncthreads();
        temp=p[threadIdx.x];
__syncthreads();
        p[threadIdx.x + 32] = temp;
__syncthreads();
        temp=p[threadIdx.x];
__syncthreads();
        p[threadIdx.x + 64] = temp;
__syncthreads();
        temp=p[threadIdx.x];
__syncthreads();
        p[threadIdx.x + 128] = temp;
__syncthreads();
        temp=p[threadIdx.x];
__syncthreads();
        p[threadIdx.x + 256] = temp;
__syncthreads();
        temp=p[threadIdx.x];
__syncthreads();
        p[threadIdx.x + 512] = temp;
__syncthreads();

        local_p[f] = p[threadIdx.x];
    }
    __syncthreads();

    if( tid >= N || tid == i || d_s[tid] == 0){
        return;
    } else {
        
        //dominates d_s[p] & d_s[tid]
        for (k=0; k<D; k++) {
            if (local_p[k] < d_P[tid*D+k]) {
                return;
            }
        }
        for (k=0; k<D; k++) {
            if (local_p[k] > d_P[tid*D+k]) {

                d_s[tid]=0;
                return;
            }
        }
    }
}


int skyline_cuda_2( const points_t *points, int *s )
{
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;
    const int nblocks=1+(N-1)/BLKDIM;
    int i, r = 0;

    for (i=0; i<N; i++) {
        s[i] = 1;
    }
    int *d_s;
    float *d_P;

    cudaSafeCall( cudaMalloc( (void**)&d_s, N*sizeof(int)) );
    cudaSafeCall( cudaMalloc( (void**)&d_P, N*D*sizeof(float)) );

    cudaSafeCall( cudaMemcpy( d_s, s, N*sizeof(int), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy( d_P, P, N*D*sizeof(float), cudaMemcpyHostToDevice) );

    for (i=0; i<N; i++) {
        if ( s[i] ) {
            kernel_2<<<nblocks,BLKDIM,2*BLKDIM*sizeof(float)>>>(i, N, D, d_s, d_P);

            cudaSafeCall( cudaMemcpy( s, d_s, N*sizeof(int), cudaMemcpyDeviceToHost) );
            cudaSafeCall( cudaMemcpy( d_s, s, N*sizeof(int), cudaMemcpyHostToDevice) );
        }
    }
    for(i=0;i<N;i++){
        if(s[i]) r++;
    }
    cudaFree(d_s);
    cudaFree(d_P);

    return r;
}

/* in questa ultima versione ho cercato di usare l'idea della versione precedente per costruire il vettore delle coordinate del punto i e unirla a quella 
di ruotare l'input per avere accessi contigui */
__global__ void kernel_3(int i, int N, int D, int *d_s, float *d_P){

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float p[];
    int k;
    float glob_temp;
    float temp;
    float local_p[MAX_D];
    //int max_index = N*D; idea brutta per evitare di mettere un operatore %
    
/* accedo in memoria globale solo 1 volta, poi sfrutto la memoria shared per costruirmi 1 vettore locale,
 che avrà valori pari alla dimensione i-esima di P */
    if(threadIdx.x < D){
        glob_temp=d_P[i+N*threadIdx.x];
    } else {
        glob_temp = 0;
    }
//__syncthreads();
    for(k=0; k<D; k++){
        
        /* ora ogni volta ricopio in memoria shared il vettore glob_temp (quello preso dalla memoria globale) spostandosi di posizione (da 0 a +1 +2 +4 +8 +16)
           ricopiando temporaneamente il vettore ottenuto ogni volta in temp e procedendo da temp */
__syncthreads(); 
        temp=glob_temp;
        p[threadIdx.x] = temp;
__syncthreads();
        glob_temp = p[threadIdx.x + 1];
__syncthreads();
        p[threadIdx.x + 1] = temp;
__syncthreads();
        temp=p[threadIdx.x];
__syncthreads();
        p[threadIdx.x + 2] = temp;
__syncthreads();
        temp=p[threadIdx.x];
__syncthreads();
        p[threadIdx.x + 4] = temp;
__syncthreads();
        temp=p[threadIdx.x];
__syncthreads();
        p[threadIdx.x + 8] = temp;
__syncthreads();
        temp=p[threadIdx.x];
__syncthreads();
        p[threadIdx.x + 16] = temp;
__syncthreads();
        temp=p[threadIdx.x];
__syncthreads();
        p[threadIdx.x + 32] = temp;
__syncthreads();
        temp=p[threadIdx.x];
__syncthreads();
        p[threadIdx.x + 64] = temp;
__syncthreads();
        temp=p[threadIdx.x];
__syncthreads();
        p[threadIdx.x + 128] = temp;
__syncthreads();
        temp=p[threadIdx.x];
__syncthreads();
        p[threadIdx.x + 256] = temp;
__syncthreads();
        temp=p[threadIdx.x];
__syncthreads();
        p[threadIdx.x + 512] = temp;
__syncthreads();

        local_p[k] = p[threadIdx.x];
    }

    __syncthreads();
    if( tid >= N || tid == i || d_s[tid] == 0){
        return;
    } else {
        
        //dominates d_s[p] & d_s[tid]
        for (k=0; k<D; k++) {
            if (local_p[k] < d_P[tid+N*k]) {
                return;
            }
        }
        for (k=0; k<D; k++) {
            if (local_p[k] > d_P[tid+N*k]) {

                d_s[tid]=0;
                return;
            }
        }
    }
}



int skyline_cuda_3( const points_t *points, int *s )
{
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;
    const int nblocks=1+(N-1)/BLKDIM;
    int i,k, r = 0;
    float *h_i;
    h_i=(float*)calloc(N*D,sizeof(float));

    for (i=0; i<N; i++) {
        s[i] = 1;
    }
    /* qua ruoto i punti x1,y1,x2,y2,x3,y3... -> x1,x2,x3,...,y1,y2,y3 */
    for(k=0; k<D; k++){
        for(i=0; i<N; i++){
            h_i[i+N*k] = P[k+i*D];
        }
    }
    int *d_s;
    float *d_i;
    
    cudaSafeCall( cudaMalloc( (void**)&d_s, N*sizeof(int)) );
    cudaSafeCall( cudaMalloc( (void**)&d_i, N*D*sizeof(float)) );

    cudaSafeCall( cudaMemcpy( d_s, s, N*sizeof(int), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy( d_i, h_i, N*D*sizeof(float), cudaMemcpyHostToDevice) );
    for (i=0; i<N; i++) {
        if ( s[i] ) {
            kernel_3<<<nblocks,BLKDIM,2*BLKDIM*sizeof(float)>>>(i, N, D, d_s, d_i);
            cudaSafeCall( cudaMemcpy( s, d_s, N*sizeof(int), cudaMemcpyDeviceToHost) );
            cudaSafeCall( cudaMemcpy( d_s, s, N*sizeof(int), cudaMemcpyHostToDevice) );
        }
    }
    for(i=0;i<N;i++){
        if(s[i]) r++;
    }
    cudaFree(d_s);
    cudaFree(d_i);
    free(h_i);
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
    /*
    const double tstart = hpc_gettime();
    const int r = skyline(&points, s);
    const double elapsed = hpc_gettime() - tstart;
    print_skyline(&points, s, r);

    fprintf(stderr,
            "\n\t%d points\n\t%d dimensione\n\t%d points in skyline\n\nExecution time %f seconds\n",
            points.N, points.D, r, elapsed);
    */

    if(points.D < 100){
        const double cuda_start_0 = hpc_gettime();
        const int cuda_r_v0 = skyline_cuda_0(&points, s);
        const double cuda_elapsed_0 = hpc_gettime() - cuda_start_0;
        print_skyline(&points, s, cuda_r_v0);

        fprintf(stderr,
                "\n(v_0)\t%d points\n\t%d dimensione\n\t%d points in skyline\n\nExecution time %f seconds\n",
                points.N, points.D, cuda_r_v0, cuda_elapsed_0);


        const double cuda_start = hpc_gettime();
        const int cuda_r = skyline_cuda_1(&points, s);
        const double cuda_elapsed = hpc_gettime() - cuda_start;
        print_skyline(&points, s, cuda_r);

        fprintf(stderr,
            "\n(v_1)\t%d points\n\t%d dimensione\n\t%d points in skyline\n\nExecution time %f seconds\n",
            points.N, points.D, cuda_r, cuda_elapsed);
    } else { 
        fprintf(stderr, "v_0 e v_1 non utilizzate poiché il numero di dimensioni troppo alto\n");
    }

    const double cuda_start_2 = hpc_gettime();
    const int cuda_r_v2 = skyline_cuda_2(&points, s);
    const double cuda_elapsed_2 = hpc_gettime() - cuda_start_2;
    print_skyline(&points, s, cuda_r_v2);

    fprintf(stderr,
            "\n(v_2)\t%d points\n\t%d dimensione\n\t%d points in skyline\n\nExecution time %f seconds\n",
            points.N, points.D, cuda_r_v2, cuda_elapsed_2);



    const double cuda_start_3 = hpc_gettime();
    const int cuda_r_v3 = skyline_cuda_3(&points, s);
    const double cuda_elapsed_3 = hpc_gettime() - cuda_start_3;
    print_skyline(&points, s, cuda_r_v3);

    fprintf(stderr,
            "\n(v_3)\t%d points\n\t%d dimensione\n\t%d points in skyline\n\nExecution time %f seconds\n",
            points.N, points.D, cuda_r_v3, cuda_elapsed_3);

    free_points(&points);
    free(s);
    return EXIT_SUCCESS;
}
