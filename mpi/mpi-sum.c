/* */
/****************************************************************************
 *
 * mpi-sum.c - Sum the content of an array using send/receive
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
 * Compile with:
 * mpicc -std=c99 -Wall -Wpedantic mpi-sum.c -o mpi-sum
 *
 * Run with:
 * mpirun -n 4 ./mpi-sum
 *
 ****************************************************************************/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* Compute the sum of all elements of array |v| of length |n| */
float sum(float *v, int n)
{
    float sum = 0;
    int i;
    
    for (i=0; i<n; i++) {
        sum += v[i];
    }
    return sum;
}

/* Fill array v of length n; store into *expected_sum the sum of the
   content of v */
void fill(float *v, int n, float *expected_sum)
{
    const float vals[] = {1, -1, 2, -2, 0};
    const int NVALS = sizeof(vals)/sizeof(vals[0]);
    int i;
    
    for (i=0; i<n; i++) {
        v[i] = vals[i % NVALS];
    }
    switch(i % NVALS) {
    case 1: *expected_sum = 1; break;
    case 3: *expected_sum = 2; break;
    default: *expected_sum = 0;
    }
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz;
    float *master_array = NULL, s, expected;
    int n = 10000;

    
    MPI_Init( &argc, &argv );	
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );
    printf("\n[1]\n");


    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    /******************/
    const int local_n = n/comm_sz;
    
    float *local_array=(float*)malloc(local_n*sizeof(float));
    
    if(0 == my_rank){
        master_array=(float*)malloc(n*sizeof(float));
        fill(master_array,n,&expected);
    }
    MPI_Scatter(master_array,   //const void *sendbuf
                local_n,        //int sendcount,
                MPI_FLOAT,      //MPI_Datatype sendtype
                local_array,    //void *recvbuf,
                local_n,        //int recvcount,
                MPI_FLOAT,      //MPI_Datatype recvtype,
                0,              //int root
                MPI_COMM_WORLD //MPI_Comm comm
                );
    
    const float local_sum = sum(local_array, local_n);
    
    free(local_array);

    if ( 0 == my_rank ) {
    
        s=sum(master_array+local_n*comm_sz,n-local_n*comm_sz);
        float remote_sum;
        s+=local_sum;
        for(int i=0; i<comm_sz-2; i++){
            

            MPI_Recv(&remote_sum, 1, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            s+=remote_sum;
        }

        s = sum(master_array, n);
    } else {
    
        MPI_Send(&local_sum, //const void *buf,
            1,        //int count,
            MPI_FLOAT,        //MPI_Datatype datatype,
            0,        //int dest,
            0,        //int tag,
            MPI_COMM_WORLD);



    }
    free(master_array);
    
    if (0 == my_rank) {
        printf("Sum=%f, expected=%f\n", s, expected);
        if (s == expected) {
            printf("Test OK\n");
        } else {
            printf("Test FAILED\n");
        }
    }
    
    MPI_Finalize();		
    return EXIT_SUCCESS;
}
