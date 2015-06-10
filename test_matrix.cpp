#include <iostream>
#include <ctime>
#include "matrix.h"
using std::cout;
using std::endl;
using std::string;
using namespace MATRIX;

int main()
{   
    clock_t start, end;
    start = clock();
    int multi = 4;
    int nrow1 = 200*multi;
    int ncol1 = 200*multi;
    int nrow2 = ncol1;
    int ncol2 = 200*multi;
    MatrixDense<double> A(nrow1, ncol1, "aaa");
    A.SetAllValue(10, "dRANDOM");
    //A.WriteMatlabDense("./data/aaa.dat");
    //A.Show(9);
    MatrixDense<double> B(nrow2, ncol2, "bbb");
    B.SetAllValue(13, "dRANDOM");
    MatrixDense<double> C = A*B;
    end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("USING time = %f\n", time);
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
    
    return 0;
}
