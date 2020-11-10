/* */
/****************************************************************************
 *
 * mpi-my-bcast.c - Broadcast using point-to-point communications
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
 * --------------------------------------------------------------------------
 *
 * Compile with
 * mpicc -std=c99 -Wall -Wpedantic mpi-my-bcast.c -o mpi-my-bcast
 *
 * run with:
 * mpirun -n 4 ./mpi-my-bcast
 *
 ****************************************************************************/
#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void my_Bcast(int *v)
{
    int my_rank, comm_sz;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);    
    if(my_rank == 0){
        //il master manda il segnale al thread 1 e 2
        if(comm_sz>1){
            MPI_Send(v,//const void *buf,
                        1,//int count,
                        MPI_INT, //MPI_Datatype datatype,
                        1,  //int dest,
                        0,
                        MPI_COMM_WORLD);
        }
        if(comm_sz>2){
            MPI_Send(v,//const void *buf,
                    1,//int count,
                    MPI_INT, //MPI_Datatype datatype,
                    2,  //int dest,
                    0,
                    MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(   v,
                    1, //int count,
                    MPI_INT, // MPI_Datatype datatype,
                    (my_rank-1)/2,
                    0, // tag
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );
        //mando il primo messaggio, se ha senso
        if(my_rank*2+1<comm_sz){
                MPI_Send(v,//const void *buf,
                        1,//int count,
                        MPI_INT, //MPI_Datatype datatype,
                        my_rank*2+1,  //int dest,
                        0,
                        MPI_COMM_WORLD
                        );
        }
    
        //mando il secondo messaggio se ha senso
        if(my_rank*2+2<comm_sz){
            MPI_Send(v,//const void *buf,
                    1,//int count,
                    MPI_INT, //MPI_Datatype datatype,
                    my_rank*2+2,  //int dest,
                    0,
                    MPI_COMM_WORLD
                    );
        }

    }
}


int main( int argc, char *argv[] )
{
    int my_rank;
    int v;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if ( 0 == my_rank ) {
        v = 999; /* only process 0 sets the value to be sent */
    } else {
        v = -1; /* all other processes set v to -1; if everything goes well, the value will be overwritten with the value received from the master */ 
    }

    my_Bcast(&v);

    if ( v == 999 ) {
        printf("OK: ");
    } else {
        printf("ERROR: ");
    }
    printf("Process %d has %d\n", my_rank, v);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
