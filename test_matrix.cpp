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
    clock_t start, end, start0, end0;
    double time;
    int multi;
    
    start0 = clock();
    MPI_Init( &argc, &argv );

    int node, total_node;
    MPI_Comm_size( MPI_COMM_WORLD, &total_node );
    MPI_Comm_rank( MPI_COMM_WORLD, &node );

    int nrow1 = atoi(argv[1]);
    int ncol1 = atoi(argv[2]);
    int nrow2 = ncol1;
    int ncol2 = atoi(argv[3]);
    int FACTOR = atoi(argv[4]);

    MatrixDense<double> A(nrow1, ncol1, "aaa");
    A.SetAllValue(10, "dRANDOM");
    //A.WriteMatlabDense("./data/aaa.dat");
    //A.Show(9);
    MatrixDense<double> B(nrow2, ncol2, "bbb");
    B.SetAllValue(13, "dRANDOM");
    //B.Show(9);
    
    start = clock();
    MatrixDense<double> C = A.MultiplyMPI(B, MPI_COMM_WORLD);
    end = clock();
    time = (double)(end - start) / CLOCKS_PER_SEC;
    if( node == 0 )
        printf("Calling MultiplyMPI() time = %f\n", time);
    
      
    if( node == 0 )
    {
	start = clock();
	MatrixDense<double> D = A*B;
	//D.Show(9);
	end = clock();
	time = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Multiply *        using time = %f\n", time);
    
	start = clock();
	MatrixDense<double> E = A.MultiplyDirect(B);
	//E.Show(9);
	end = clock();
	time = (double)(end - start) / CLOCKS_PER_SEC;
	printf("MultiplyDirect    using time = %f\n", time);
    
	start = clock();
	MatrixDense<double> B_T = B.Transform();
	//B_T.Show(9);
	MatrixDense<double> F = A.MultiplyTransform(B_T, FACTOR);
	//F.Show(9);
	end = clock();
	time = (double)(end - start) / CLOCKS_PER_SEC;
        printf("MultiplyTransform using time = %f\n", time);
        
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
    
    /*
    MatrixDense<int> A100x100_1;
    A100x100_1.ReadMatlabDense("./data/imat100x100_1.dat");
    A100x100_1.SetName("A100x100_1");
    MatrixDense<int> A100x100_2;
    A100x100_2.ReadMatlabDense("./data/imat100x100_2.dat");
    A100x100_2.SetName("A100x100_2");
    MatrixDense<int> M;
    //M = A100x100_1.MultiplySlice( 2, 10, 7, 15, A100x100_2, 22, 30, 35, 41 );
    M.SetName("multi_slice");
    //M.Show(9);
    MatrixDense<int> N;
    //N = A100x100_1.MultiplySliceTransform( 2, 10,  7, 15, A100x100_2, 35, 41, 22, 30 );
    N.SetName("multi_slice_transform");
    
    start = clock();
    //M = A100x100_1.MultiplyDirect(A100x100_2);
    end = clock();
    time = (double)(end - start) / CLOCKS_PER_SEC;
    //printf("MultiplyDirect using time = %f\n", time);
    
    start = clock();
    //M = A100x100_1.MultiplyTransform(A100x100_2);
    end = clock();
    time = (double)(end - start) / CLOCKS_PER_SEC;
    //printf("MultiplyTransform using time = %f\n", time);
    */

    /*
    MatrixDense<int> A10x10_1;
    A10x10_1.ReadMatlabDense("./data/imat10x10_1.dat");
    A10x10_1.SetName("A10x10_1");
    A10x10_1.Show(9);
    MatrixDense<int> A10x10_2;
    A10x10_2.ReadMatlabDense("./data/imat10x10_2.dat");
    A10x10_2.SetName("A10x10_2");
    A10x10_2.Show(9);
    MatrixDense<int> M10x10 = A10x10_1 * A10x10_2;
    M10x10.SetName("M10x10");
    M10x10.Show(9);
    M10x10.WriteMatlabDense("./data/imatM10x10.dat");
    */
    MPI_Finalize();
    
    end0 = clock();
    time = (double)(end0 - start0) / CLOCKS_PER_SEC;
    if( node == 0 )
        printf("main time = %f\n", time);
        
    return 0;
}
