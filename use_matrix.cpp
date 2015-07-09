#include <iostream>
#include <cstdlib>
#include <ctime>
#include "mpi.h"
#include "matrix.h"
using std::cout;
using std::cin;
using std::endl;
using std::string;
using namespace MATRIX;

int main( int argc, char *argv[] )
{
    clock_t start_main, end_main;
    double start, end;
    start_main = clock();
    
    int node, total_node;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &total_node );
    MPI_Comm_rank( MPI_COMM_WORLD, &node );

    int nrow1 = atoi(argv[1]);
    int ncol1 = atoi(argv[2]);
    int nrow2 = ncol1;
    int ncol2 = atoi(argv[3]);
    int FACTOR = atoi(argv[4]);

    MatrixDense<double> A(nrow1, ncol1, "A");
    A.SetAllValue(10, "dRANDOM");
    
    MatrixDense<double> B(nrow2, ncol2, "B");
    B.SetAllValue(13, "dRANDOM");
    
    start = MPI_Wtime();
    MatrixDense<double> C = A.MultiplyMPI(B, MPI_COMM_WORLD);
    end = MPI_Wtime();
    if( node == 0 )
        printf("Calling MultiplyMPI() time = %f\n", end-start);
      
    if( node == 0 )
    {
	start = MPI_Wtime();
	MatrixDense<double> D = A*B;
	end = MPI_Wtime();
	printf("Multiply *        using time  = %f\n", end-start);
    
	start = MPI_Wtime();
	MatrixDense<double> E = A.MultiplyDirect(B);
	end = MPI_Wtime();
	printf("MultiplyDirect    using time  = %f\n", end-start);
    
	start = MPI_Wtime();
	MatrixDense<double> B_T = B.Transform();
	MatrixDense<double> F = A.MultiplyTransform(B_T, FACTOR);
	end = MPI_Wtime();
        printf("MultiplyTransform using time  = %f\n", end-start);
        
        cout << "check result: ";
        MatrixDense<double> W = C-D;
        cout << W.IsZero(0.00000001);
        
        W = D-E;
        cout << ", " << W.IsZero(0.00000001);

        W = E-F;
        cout << ", " << W.IsZero(0.00000001);

        W = F-C;
        cout << ", " << W.IsZero(0.00000001);
	cout << endl;
    }
    MPI_Finalize();
    
    end_main = clock();
    if( node == 0 )
        printf("main time = %f\n", (double)(end_main-start_main)/CLOCKS_PER_SEC);
        
    return 0;
}
