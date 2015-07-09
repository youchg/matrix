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
    clock_t start, end;
    double  time;
    MatrixDense<int> A100x100_1;
    A100x100_1.ReadMatlabDense("./data/imat100x100_1.dat");
    A100x100_1.SetName("A100x100_1");
    MatrixDense<int> A100x100_2;
    A100x100_2.ReadMatlabDense("./data/imat100x100_2.dat");
    A100x100_2.SetName("A100x100_2");
    MatrixDense<int> M;
    M = A100x100_1.MultiplySlice( 2, 10, 7, 15, A100x100_2, 22, 30, 35, 41 );
    M.SetName("multi_slice");
    M.Show(9);
    MatrixDense<int> N;
    N = A100x100_1.MultiplySliceTransform( 2, 10, 7, 15, A100x100_2, 35, 41, 22, 30 );
    N.SetName("multi_slice_transform");
    
    start = clock();
    M = A100x100_1.MultiplyDirect(A100x100_2);
    end = clock();
    time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("MultiplyDirect using time = %f\n", time);
    
    start = clock();
    M = A100x100_1.MultiplyTransform(A100x100_2);
    end = clock();
    time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("MultiplyTransform using time = %f\n", time);

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
     
    return 0;
}
