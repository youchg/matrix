// matrix.h  -- matrix in dense and CSR format
#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <cstring>// memcpy
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include "mpi.h"

#define PRINT_ALL  9
#define PRINT_MOST 5
#define PRINT_FEW  1
#define PRINT_NONE 0

#define DEBUG_MODE 0
// DEBUG_MODE 代表 DEBUG 等级，与上面打印等级比较后打印相应信息

namespace MATRIX
{
using std::cout;
using std::string;
using std::endl;
using std::ofstream;
using std::ifstream;

class MatrixBase
{
private:
    string name;
    int  nrow;
    int  ncol;

public:
#if DEBUG_MODE > PRINT_ALL
    MatrixBase(const int nr=0, 
	       const int nc=0,
	       const string str="noname");
    virtual ~MatrixBase();
#else
    MatrixBase(const int nr=0, 
	       const int nc=0,
	       const string str="noname")
	       : name(str), nrow(nr), ncol(nc) {}
    virtual ~MatrixBase(){}
#endif
    int GetNRow(void)const { return nrow; }
    int GetNCol(void)const { return ncol; }
    string GetName(void)const { return name; }

    void SetNRow(int nr){ nrow = nr; }
    void SetNCol(int nc){ ncol = nc; }
    void SetName(string na){ name = na; }
   
    virtual void Show(const int print_level=PRINT_FEW)const = 0;
    //virtual void Write(const string filename, 
	               //const string format)const = 0;
};

template <typename Type>
class MatrixDense: public MatrixBase
{
private:
    Type **val;

public:
    MatrixDense(const int nr=0, const int nc=0,
                const string str="noname", 
	        const Type **va=0);
    ~MatrixDense();
    MatrixDense(const MatrixDense &B);

    void SetAllValue( const Type va, const string mode="CONSTANT" );// CONSTANT RANDOM
    //还可以增加参数，使得能生成 DIAGONAL TRIDIAGONAL 等类型，
    //但对于满矩阵处理来说，并不针对这些做优化
    virtual void Show (const int print_level=PRINT_FEW)const;
    //virtual void Write(const string filename, 
	               //const string format)const;

    MatrixDense & operator=(const MatrixDense &B); // A = B
    MatrixDense   operator+(const MatrixDense &B)const; // A + B
    MatrixDense   operator-(const MatrixDense &B)const; // A - B
    MatrixDense   operator*(const MatrixDense &B)const; // A * B

    MatrixDense   operator*(const Type alpha)const; // A * alpha 
    template <typename type>
    friend MatrixDense<type> operator*(const type alpha, const MatrixDense<type> &A); // alpha * A

public:
    void WritePS(const string filename)const; 
    void WriteMatlabDense(const string filename)const; 
    void ReadMatlabDense(const string filename);
    MatrixDense MultiplySlice(const int row_begin, const int row_end,
                              const int col_begin, const int col_end,
                              const MatrixDense &B,
                              const int B_row_begin, const int B_row_end,
                              const int B_col_begin, const int B_col_end)const;
    MatrixDense MultiplySliceTransform(
                              const int row_begin, const int row_end,
                              const int col_begin, const int col_end,
                              const MatrixDense &B,
                              const int B_row_begin, const int B_row_end,
                              const int B_col_begin, const int B_col_end)const;
    MatrixDense MultiplyDirect(const MatrixDense &B)const; // A*B
    MatrixDense MultiplyTransform(const MatrixDense &B)const; // A*B_T
    MatrixDense MultiplyMPI(const MatrixDense &B, const MPI_Comm comm)const; // A*B
    MatrixDense Transform()const;
    MatrixDense Sub(const int row_begin, const int row_end,
                    const int col_begin, const int col_end)const;
};

#if 0
export template <typename Type>
class MatrixCSR: public MatrixBase
{
private:
    int  nnz;

    Type *aa;
    int  *ja;
    int  *ia;

public:
    MatrixCSR(const string str="empty", 
	      const int nr=0, const int nc=0, 
	      const int nz=0,  
	      const Type *aa=0, const int *ja=0, const int *ia=0);
    ~MatrixCSR();
    MatrixCSR(const MatrixCSR &B);

    void Show(const int print_level = 0);
    void Draw(const string filename);
    void Write4Matlab(const string filename);

    virtual void Show (const int print_level)const;
    virtual void Write(const string filename, 
	               const string format)const;

    MatrixCSR & operator=(const MatrixCSR &B); // A = B
    MatrixCSR   operator+(const MatrixCSR &B)const; // A + B
    MatrixCSR   operator-(const MatrixCSR &B)const; // A - B
    MatrixCSR   operator*(const MatrixCSR &B)const; // A * B

    MatrixCSR   operator*(const double alpha)const; // A * alpha 
    friend MatrixCSR operator*(const double alpha, const MatrixCSR &A); // alpha * A
}
#endif



#if DEBUG_MODE > PRINT_ALL
//此时需要打印函数调用信息，所以不能使用内联函数
//否则直接在类里面定义即可
MatrixBase::MatrixBase( const int nr, const int nc,
                        const string str)
                        : name(str), nrow(nr), ncol(nc)
{
    cout << name << ": calling MatrixBase constructor." << endl;
}

MatrixBase::~MatrixBase()
{
    cout << name << ": calling MatrixBase destructor." << endl;
}
#endif

template <typename Type>
MatrixDense<Type>::MatrixDense(const int nr, const int nc, 
	                 const string str,
	                 const Type **va)
	                 : MatrixBase(nr, nc,str)
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() << ": calling MatrixDense constructor." << endl;
#endif

    if( nr!=0 && nc!=0 )
    {
	val = new Type* [nr];
	for( int i(0); i<nr; i++ )
	    val[i] = new Type [nc];

        if( va!=0 )
	{
	    for( int i(0); i<nr; i++ )
		for( int j(0); j<nc; j++ )
		    val[i][j] = va[i][j];
	}
	else
	{
	    for( int i(0); i<nr; i++ )
		for( int j(0); j<nc; j++ )
		    val[i][j] = 0;
	}
    }
    else
    {
	val = 0;
    }

}

template <typename Type>
MatrixDense<Type>::~MatrixDense()
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() << ": calling MatrixDense destructor." << endl;
#endif

    int nrow = MatrixBase::GetNRow();
    int ncol = MatrixBase::GetNCol();
    if( nrow!=0 && ncol!=0 )
    {
	for( int i(0); i<nrow; i++ )
	    delete [] val[i];
    
	delete [] val;
    }
    val = 0;
}

template <typename Type>
void MatrixDense<Type>::SetAllValue( const Type va, const string mode )
//CONSTANT: 所有值设为 va
//iRANDOM : 随机产生 int 类型的值，范围在 0-va 之间
//dRANDOM : 随机产生 double 类型的值，范围在 0-va 之间
//待处理  : va 与 0 的比较
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() << ": calling MatrixDense SetAllValue." << endl;
#endif

    int nrow = MatrixBase::GetNRow();
    int ncol = MatrixBase::GetNCol();
    if( nrow!=0 && ncol!=0 )
    {
        if( mode=="iRANDOM" )
        {
            srand(24);
            for( int i(0); i<nrow; i++ )
	        for( int j(0); j<ncol; j++ )
		    val[i][j] = rand()%((int)va);
        }
        else if( mode=="dRANDOM" )
        {
            srand(24);
            //srand( (unsigned)time(0) );
            for( int i(0); i<nrow; i++ )
	        for( int j(0); j<ncol; j++ )
		    val[i][j] = (Type)( rand() / ((double)RAND_MAX/va) );
        }
        else
        {
	    for( int i(0); i<nrow; i++ )
	        for( int j(0); j<ncol; j++ )
		    val[i][j] = va;
        }        
    }
}

template <typename Type>
MatrixDense<Type>::MatrixDense(const MatrixDense<Type> &B)
                              : MatrixBase(B)
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() << ": calling MatrixDense copy constructor." 
         << endl;
#endif

    int nrow = MatrixBase::GetNRow();
    int ncol = MatrixBase::GetNCol();
    if( nrow!=0 && ncol!=0 )
    {    
        val = new Type* [nrow];
        for( int i(0); i<nrow; i++ )
        {
            val[i] = new Type [ncol];
            if( B.val != 0 )
	    {
                memcpy( val[i], B.val[i], ncol*sizeof(Type) );
	    }
        }
/*
//2015.6.6 ycg 注释此块
        if( B.val!=0 )
	{
            for( int i(0); i<nrow; i++ )
	    {
                memcpy( val[i], B.val[i], ncol*sizeof(Type) );
	    }
	}
*/
    }
    else
    {
        val = 0;
    }
}

template <typename Type>
MatrixDense<Type> & MatrixDense<Type>::operator=(const MatrixDense<Type> &B)
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() << ": calling MatrixDense::operator=." 
         << endl;
#endif
    if( this == &B )
        return *this;

    int nrow, ncol;
    nrow = MatrixBase::GetNRow();
    ncol = MatrixBase::GetNCol();
    if( nrow!=0 && ncol!=0 )
    {
	for( int i(0); i<nrow; i++ )
	    delete [] val[i];
	delete [] val;
    }

    MatrixBase::operator=(B);
    nrow = MatrixBase::GetNRow();
    ncol = MatrixBase::GetNCol();
    if( nrow!=0 && ncol!=0 )
    {
	val = new Type* [nrow];
	for( int i(0); i<nrow; i++ )
	    val[i] = new Type [ncol];
        
        if( B.val!=0 )
        {
	    for( int i(0); i<nrow; i++ )
                for( int j(0); j<ncol; j++ )
	            val[i][j] = B.val[i][j];
	        //memcpy( val[i], B.val[i], ncol*sizeof(Type) );
	}
    }
    else
    {
	val = 0;
    }

    return *this;
}

template <typename Type>
void MatrixDense<Type>::Show(const int print_level)const
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() << ": calling MatrixDense::Show(...)." 
         << endl;
#endif
    if( print_level>PRINT_NONE )
    {
        cout << "==============================" << endl
             << MatrixBase::GetName() << ": "
             << "nrow = " << MatrixBase::GetNRow() << ", "
	     << "ncol = " << MatrixBase::GetNCol() << endl;
    }
    if( print_level>PRINT_MOST )
    {
	if( val==0 )
	{
	    cout << "ERROR--MatrixDense::Show : val point to NULL!" << endl;
	}
	else
	{
	    for( int i(0); i<MatrixBase::GetNRow(); i++ )
	    {
	        for( int j(0); j<MatrixBase::GetNCol(); j++ )
		{
		    cout << i+1 << " " << j+1 << " "
			 << val[i][j] << endl;
		}
	    }
	}
    }
    if( print_level>PRINT_NONE )
    {
        cout << "==============================" << endl;
    }

}

template <typename Type>
MatrixDense<Type> MatrixDense<Type>::operator+(const MatrixDense<Type> &B)const
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() << ": calling MatrixDense::operator+" 
         << endl;
#endif
    int nrow = MatrixBase::GetNRow();
    int ncol = MatrixBase::GetNCol();
    if( nrow!=B.GetNRow() || ncol!=B.GetNCol() )
    {
	cout << "ERROR in MatrixDense::operator+ : Matrix dimensions must agree." << endl;
	return MatrixDense<Type>();
    }
    
    MatrixDense<Type> C(nrow, ncol,"SUM");
    if( nrow!=0 && ncol!=0 )
    {
	for( int i(0); i<nrow; i++ )
	{
	    for( int j(0); j<ncol; j++ )
	    {
		C.val[i][j] = val[i][j] + B.val[i][j];
	    }
	}
    }
    return C;
}

template <typename Type>
MatrixDense<Type> MatrixDense<Type>::operator-(const MatrixDense<Type> &B)const
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() << ": calling MatrixDense::operator-" 
         << endl;
#endif
    int nrow = MatrixBase::GetNRow();
    int ncol = MatrixBase::GetNCol();
    if( nrow!=B.GetNRow() || ncol!=B.GetNCol() )
    {
	cout << "ERROR in MatrixDense::operator- : Matrix dimensions must agree." << endl;
	return MatrixDense<Type>();
    }
    
    MatrixDense<Type> C(nrow, ncol,"DIFF");
    if( nrow!=0 && ncol!=0 )
    {
	for( int i(0); i<nrow; i++ )
	{
	    for( int j(0); j<ncol; j++ )
	    {
		C.val[i][j] = val[i][j] - B.val[i][j];
	    }
	}
    }
    return C;
}

template <typename Type>
MatrixDense<Type> MatrixDense<Type>::operator*(const Type alpha)const
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() << ": calling MatrixDense::operator*" 
         << endl;
#endif
    int nrow = MatrixBase::GetNRow();
    int ncol = MatrixBase::GetNCol();
    
    MatrixDense<Type> C(nrow, ncol, "PRODUCT");
    if( nrow!=0 && ncol!=0 )
    {
	for( int i(0); i<nrow; i++ )
	{
	    for( int j(0); j<ncol; j++ )
	    {
		C.val[i][j] = val[i][j] * alpha;
	    }
	}
    }
    return C;
}

template <typename type>
MatrixDense<type> operator*(const type alpha, const MatrixDense<type> &A)
{
#if DEBUG_MODE > PRINT_ALL
    cout << A.GetName() << ": calling friend MatrixDense::operator*" 
         << endl;
#endif
    return A*alpha;
}

template <typename Type>
MatrixDense<Type>  MatrixDense<Type>::operator*(const MatrixDense<Type> &B)const
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() << ": calling MatrixDense::operator*" 
         << endl;
#endif
    int nrow1 = MatrixBase::GetNRow();
    int ncol1 = MatrixBase::GetNCol();
    int nrow2 = B.GetNRow();
    int ncol2 = B.GetNCol();

    if( ncol1 != nrow2 ) 
    {
	cout << "ERROR in MatrixDense::operator* : Matrix dimensions must agree." << endl;
	return MatrixDense<Type>();
    }

    clock_t start, end;
    start = clock();
    
    MatrixDense<Type> C(nrow1, ncol2, "PRODUCT");
    if( nrow1!=0 && ncol1!=0 && ncol2!=0 )
    {
        //printf("Progress: ");
        for( int i(0); i<nrow1; i++ )
        {
            //if( i%(nrow1/10) == 0 )
                //printf("%d%%, ", i*100/nrow1);
	    for( int j(0); j<ncol2; j++ )
	    {
	        for( int k(0); k<ncol1; k++ )
	        {
		    C.val[i][j] += val[i][k] * B.val[k][j];
		    //C.val[i][j] += val[i][k] * B.val[j][k];
	        }
	    }
        }
        //printf("\n");
    }
    end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("MatrixDense::operator*---time = %f\n", time);

    return C;
}

/*
template <typename Type>
void MatrixDense<Type>::WritePS(const string filename)const
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() 
	 << ": calling MatrixDense::WritePS(...)" << endl;
#endif

}
*/

template <typename Type>
void MatrixDense<Type>::WriteMatlabDense(const string filename)const
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() 
	 << ": calling MatrixDense::WriteMatlabDense(...)" << endl;
#endif
    ofstream fout(filename.c_str());
    if( !fout.is_open() )
    {
	cout << "Error in MatrixDense::WriteMatlabDense: cannot open \"" 
	     << filename << "\"!" << endl;
	return;
    }
    
    int nrow = MatrixBase::GetNRow(); 
    int ncol = MatrixBase::GetNCol(); 
    fout << nrow << endl;
    fout << ncol << endl;
    fout << nrow*ncol << endl;

    for( int i(0); i<nrow; i++ )
    {
	for( int j(0); j<ncol; j++ )
	{
	    fout << i+1 << " " << j+1 << " "
		 << val[i][j] << endl;
	}
    }
    fout.close();
}

template <typename Type>
void MatrixDense<Type>::ReadMatlabDense(const string filename)
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() 
	 << ": calling MatrixDense::ReadMatlabDense(...)" << endl;
#endif
    ifstream fin(filename.c_str());
    if( !fin.is_open() )
    {
	cout << "Error in MatrixDense::ReadMatlabDense: cannot open \"" 
	     << filename << "\"!" << endl;
	return;
    }

    int nrow, ncol, nnz;
    fin >> nrow >> ncol >> nnz;

    if( nnz != nrow*ncol )
    {
	cout << "ERROR in MatrixDense::ReadMatlabDense: \""
	     << filename << "\" is not a dense matrix format file."
	     << endl;
	return; 
    }

    if( val!=0 )
    {
	for( int i(0); i<MatrixBase::GetNRow(); i++ )
	    delete [] val[i];
	delete [] val;
    }

    MatrixBase::SetNRow(nrow);
    MatrixBase::SetNCol(ncol);
    val = new Type* [nrow];
    for( int i(0); i<nrow; i++ )
	val[i] = new Type [ncol];

    int row,col;
    for( int i(0); i<nnz; i++ )
    {
	fin >> row >> col;
	fin >> val[row-1][col-1];
    }
    fin.close();
}

template <typename Type>
MatrixDense<Type> MatrixDense<Type>::MultiplySlice(
                         const int row_begin, const int row_end,
                         const int col_begin, const int col_end,
                         const MatrixDense &B,
                         const int B_row_begin, const int B_row_end,
                         const int B_col_begin, const int B_col_end)const
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() 
	 << ": calling MatrixDense::MultiplySlice(...)" << endl;
#endif
    assert( row_begin>=0 && row_end<MatrixBase::GetNRow() && 
            col_begin>=0 && col_end<MatrixBase::GetNCol() &&
            B_row_begin>=0 && B_row_end<B.GetNRow() && 
            B_col_begin>=0 && B_col_end<B.GetNCol() &&
            row_end>=row_begin && col_end>=col_begin &&
            B_row_end>=B_row_begin && B_col_end>=B_col_begin);
            
    int nrow1 = row_end - row_begin + 1;
    int ncol1 = col_end - col_begin + 1;
    int nrow2 = B_row_end - B_row_begin + 1;
    int ncol2 = B_col_end - B_col_begin + 1;
    

    if( ncol1 != nrow2 ) 
    {
	cout << "ERROR in MatrixDense::MultiplySlice : Matrix dimensions must agree." << endl;
	return MatrixDense<Type>();
    }

    clock_t start, end;
    start = clock();
    
    MatrixDense<Type> C(nrow1, ncol2, "PRODUCT");
    if( nrow1!=0 && ncol1!=0 && ncol2!=0 )
    {
        //printf("Progress: ");
        for( int i(0); i<nrow1; i++ )
        {
	    for( int j(0); j<ncol2; j++ )
	    {
	        for( int k(0); k<ncol1; k++ )
	        {
		    C.val[i][j] += val[i+row_begin][k+col_begin] * B.val[k+B_row_begin][j+B_col_begin];
		    //C.val[i][j] += val[i][k] * B.val[j][k];
	        }
	    }
        }
        //printf("\n");
    }
    end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("MatrixDense::MultiplySlice---time = %f\n", time);

    return C;
}

template <typename Type>
MatrixDense<Type> MatrixDense<Type>::MultiplySliceTransform(
                              const int row_begin, const int row_end,
                              const int col_begin, const int col_end,
                              const MatrixDense &B,
                              const int B_row_begin, const int B_row_end,
                              const int B_col_begin, const int B_col_end)const
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() 
	 << ": calling MatrixDense::MultiplySliceTransform(...)" << endl;
#endif
    assert( row_begin>=0 && row_end<MatrixBase::GetNRow() && 
            col_begin>=0 && col_end<MatrixBase::GetNCol() &&
            B_row_begin>=0 && B_row_end<B.GetNRow() && 
            B_col_begin>=0 && B_col_end<B.GetNCol() &&
            row_end>=row_begin && col_end>=col_begin &&
            B_row_end>=B_row_begin && B_col_end>=B_col_begin);
            
    int nrow1 = row_end - row_begin + 1;
    int ncol1 = col_end - col_begin + 1;
    int nrow2 = B_row_end - B_row_begin + 1;
    int ncol2 = B_col_end - B_col_begin + 1;
    

    if( ncol1 != ncol2 ) 
    {
	cout << "ERROR in MatrixDense::MultiplySliceTransform : Matrix dimensions must agree." << endl;
	return MatrixDense<Type>();
    }

    clock_t start, end;
    start = clock();
    
    MatrixDense<Type> C(nrow1, nrow2, "PRODUCT");
    if( nrow1!=0 && ncol1!=0 && nrow2!=0 )
    {
        //printf("Progress: ");
        for( int i(0); i<nrow1; i++ )
        {
	    for( int j(0); j<nrow2; j++ )
	    {
	        for( int k(0); k<ncol1; k++ )
	        {
		    C.val[i][j] += val[i+row_begin][k+col_begin] * B.val[j+B_row_begin][k+B_col_begin];
	        }
	    }
        }
        //printf("\n");
    }
    end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("MatrixDense::MultiplySliceTransform---time = %f\n", time);

    return C;
}




template <typename Type>
MatrixDense<Type> MatrixDense<Type>::MultiplyDirect(const MatrixDense &B)const
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() 
	 << ": calling MatrixDense::MultiplyDirect(...)" << endl;
#endif
    int nrow1 = MatrixBase::GetNRow();
    int ncol1 = MatrixBase::GetNCol();
    int nrow2 = B.GetNRow();
    int ncol2 = B.GetNCol();

    if( ncol1 != nrow2 ) 
    {
	cout << "ERROR in MatrixDense::MultiplyDirect : Matrix dimensions must agree." << endl;
	return MatrixDense<Type>();
    }
    
    return MultiplySlice(   0, nrow1-1, 0, ncol1-1,
                    B, 0, nrow2-1, 0, ncol2-1);
}
/*
MatrixDense MultiplyTransform(const MatrixDense &B)const; // A*B_T
MatrixDense MultiplyMPI(const MatrixDense &B, const MPI_Comm comm)const; // A*B
MatrixDense Transform()const;
MatrixDense Sub(const int row_begin, const int row_end,
                    const int col_begin, const int col_end)const;
*/
}
#endif
