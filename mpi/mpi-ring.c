/* */
/****************************************************************************
 *
 * mpi-ring.c - Ring communication with MPI
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
 * Compile with:
 * mpicc -std=c99 -Wall -Wpedantic mpi-ring.c -o mpi-ring
 *
 * Run with:
 * mpirun -n 4 ./mpi-ring
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz, K , value = 0;
    
    MPI_Init( &argc, &argv );	
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    if ( argc > 1 ) {
        K = atoi(argv[1]);
    } else K = 4; //default number of "giri" to do
    
    while(K){
	    
    	if(0==my_rank){
    		value++;
		    MPI_Send(&value, 1, MPI_INT, 1,
	    	        0, MPI_COMM_WORLD);
		    K--;

		    MPI_Recv(&value, 1, MPI_INT,
            comm_sz-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		} else {
			MPI_Recv(&value, 1, MPI_INT,
            my_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			value++;
			MPI_Send(&value,1,MPI_INT,(my_rank+1)%comm_sz,0,MPI_COMM_WORLD);
			K--; 	//in realtà penso che potrei anche evitare di "finire il while" per i processi che non sono il master e lasciarli bloccati sulla Recv ma mi pare brutto
		}
	}
	if(my_rank == 0) printf("valore finale dopo tutti i giri:%d\n",value);
    MPI_Finalize();		
    return EXIT_SUCCESS;
}
