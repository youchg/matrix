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
                              const int B_col_begin, const int B_col_end,
			      const int FACTOR = 1)const;
    MatrixDense MultiplyDirect(const MatrixDense &B)const; // A*B
    MatrixDense MultiplyTransform(const MatrixDense &B, 
                                  const int FACTOR = 1)const; // A*B_T
    MatrixDense MultiplyMPI(const MatrixDense &B, const MPI_Comm comm)const; // A*B
    MatrixDense Transform()const;
    MatrixDense Sub(const int row_begin, const int row_end,
                    const int col_begin, const int col_end)const;
    int IsZero(const Type tol = (Type)0.0000000001) const;
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
	    {
		//memset( val[i], 0, nc*sizeof(Type) );
		for( int j(0); j<nc; j++ )
		    val[i][j] = 0.0;
	    }
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
        Type tmp;
        for( int i(0); i<nrow1; i++ )
        {
	    for( int j(0); j<ncol2; j++ )
	    {
	        tmp = 0;
	        for( int k(0); k<ncol1; k++ )
	        {
		    tmp += val[i][k] * B.val[k][j];
		    //C.val[i][j] += val[i][k] * B.val[k][j];
	        }
	        C.val[i][j] = tmp;
	    }
        }
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

#if DEBUG_MODE > PRINT_FEW
    clock_t start, end;
    start = clock();
#endif
    
    MatrixDense<Type> C(nrow1, ncol2, "PRODUCT");
    if( nrow1!=0 && ncol1!=0 && ncol2!=0 )
    {
        Type tmp;
        for( int i(0); i<nrow1; i++ )
        {
	    for( int j(0); j<ncol2; j++ )
	    {
	        tmp = 0;
	        for( int k(0); k<ncol1; k++ )
	        {
		    tmp += val[i+row_begin][k+col_begin] * B.val[k+B_row_begin][j+B_col_begin];
	        }
	        C.val[i][j] = tmp;
	    }
        }
    }
    
#if DEBUG_MODE > PRINT_FEW
    end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("MatrixDense::MultiplySlice---time = %f\n", time);
#endif

    return C;
}

#define MST_MODE 2

template <typename Type>
MatrixDense<Type> MatrixDense<Type>::MultiplySliceTransform(
                              const int row_begin, const int row_end,
                              const int col_begin, const int col_end,
                              const MatrixDense &B,
                              const int B_row_begin, const int B_row_end,
                              const int B_col_begin, const int B_col_end,
			      const int FACTOR)const
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

#if DEBUG_MODE > PRINT_FEW
    clock_t start, end;
    start = clock();
#endif
    
    MatrixDense<Type> C(nrow1, nrow2, "PRODUCT");
    //printf("nrow1 = %d, ncol1 = %d, nrow2 = %d, ncol2 = %d \n", nrow1, ncol1, nrow2, ncol2);
    if( nrow1!=0 && ncol1!=0 && nrow2!=0 )
    {
        /*
        Type tmp;
        for( int i(0); i<nrow1; i++ )
        {
	    for( int j(0); j<nrow2; j++ )
	    {
	        tmp = 0;
	        for( int k(0); k<ncol1; k++ )
	        {
		    tmp += val[i+row_begin][k+col_begin] * B.val[j+B_row_begin][k+B_col_begin];
		    //C.val[i][j] += val[i+row_begin][k+col_begin] * B.val[j+B_row_begin][k+B_col_begin];
	        }
	        C.val[i][j] = tmp;
	    }
        }
        */
        Type ** Aval = val + row_begin;
        Type ** Bval = B.val + B_row_begin;
        Type ** Cval = C.val;
        Type tmp;
        
        if( col_begin==0 && B_col_begin==0 )
        {
#if MST_MODE == 1
            for( int i(0); i<nrow1; i++ )
            {
	        for( int j(0); j<nrow2; j++ )
	        {
	            tmp = 0;
	            for( int k(0); k<ncol1; k++ )
	            {
	    	        tmp += Aval[i][k] * Bval[j][k];
	            }
	            Cval[i][j] = tmp;
	        }
            }
#else
//#define FACTOR   2
	    int i(0), j(0), k(0), it(0), jt(0), kt(0);
	    int itF, jtF, ktF;
	    for( it=0; it<nrow1; it+=FACTOR )
	    {
		itF = it+FACTOR;
		for( jt=0; jt<nrow2; jt+=FACTOR )
		{
		    jtF = jt+FACTOR;
		    for( kt=0; kt<ncol1; kt+=FACTOR)
		    {
			ktF = kt+FACTOR;
			for( i=it; i<itF; i++)
			{
			    for( j=jt; j<jtF; j++ )
			    {
				tmp = (Type)0.0;
				for( k=kt; k<ktF; k++ )
				{
				    tmp += Aval[i][k] * Bval[j][k];
				    //Cval[i][j] += Aval[i][k] * Bval[j][k];
				}
				Cval[i][j] += tmp;
			    }
			}
		    }
		}
	    }
#endif
        }
        else
        {
            for( int i(0); i<=nrow1; i++ )
            {
	        for( int j(0); j<=nrow2; j++ )
	        {
	            tmp = 0;
	            for( int k(0); k<ncol1; k++ )
	            {
	    	        tmp += Aval[i][k+col_begin] * Bval[j][k+B_col_begin];
	            }
	            Cval[i][j] = tmp;
	        }
            }
        }
    }
    
#if DEBUG_MODE > PRINT_FEW
    end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("MatrixDense::MultiplySliceTransform---time = %f\n", time);
#endif

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

template <typename Type>
MatrixDense<Type> MatrixDense<Type>::MultiplyTransform(const MatrixDense &B,
	                                               const int FACTOR)const
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() 
	 << ": calling MatrixDense::MultiplyTransform(...)" << endl;
#endif
    int nrow1 = MatrixBase::GetNRow();
    int ncol1 = MatrixBase::GetNCol();
    int nrow2 = B.GetNRow();
    int ncol2 = B.GetNCol();

    if( ncol1 != ncol2 ) 
    {
	cout << "ERROR in MatrixDense::MultiplyTransform : Matrix dimensions must agree." << endl;
	return MatrixDense<Type>();
    }
    
    return MultiplySliceTransform(   0, nrow1-1, 0, ncol1-1,
                                  B, 0, nrow2-1, 0, ncol2-1, FACTOR);
}

template <typename Type>
MatrixDense<Type> MatrixDense<Type>::Transform()const
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() 
	 << ": calling MatrixDense::Transform()" << endl;
#endif
    int nrow = MatrixBase::GetNRow();
    int ncol = MatrixBase::GetNCol();

    MatrixDense<Type> A_T(ncol, nrow, "TRANSFORM");
    if( nrow!=0 && ncol!=0 )
    {
        if( val!=0 )
        {
	    for( int i(0); i<nrow; i++ )
                for( int j(0); j<ncol; j++ )
	            A_T.val[j][i] = val[i][j];
	}
    }
    else
    {
	A_T.val = 0;
    }
    
    return A_T; 
}


template <typename Type>
MatrixDense<Type> MatrixDense<Type>::Sub(
                    const int row_begin, const int row_end,
                    const int col_begin, const int col_end)const
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() 
	 << ": calling MatrixDense::Sub(...)" << endl;
#endif
    int nrow = MatrixBase::GetNRow();
    int ncol = MatrixBase::GetNCol();
    assert( row_begin>=0 && row_end<nrow && 
            col_begin>=0 && col_end<ncol &&
            row_end>=row_begin && col_end>=col_begin);
            
    int nrow1 = row_end - row_begin + 1;
    int ncol1 = col_end - col_begin + 1;

#if DEBUG_MODE > PRINT_FEW    
    clock_t start, end;
    start = clock();
#endif
    
    MatrixDense<Type> S(nrow1, ncol1, "SUB");
    if( nrow1!=0 && ncol1!=0 )
    {
        for( int i(0); i<nrow1; i++ )
        {
	    for( int j(0); j<ncol1; j++ )
	    {
		S.val[i][j] = val[i+row_begin][j+col_begin];
	    }
        }
    }

#if DEBUG_MODE > PRINT_FEW
    end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("MatrixDense::Sub---time = %f\n", time);
#endif

    return S;
}

template <typename Type>
MatrixDense<Type> MatrixDense<Type>::MultiplyMPI(const MatrixDense &B, const MPI_Comm comm)const
{
#if DEBUG_MODE > PRINT_ALL
    cout << MatrixBase::GetName() 
	 << ": calling MatrixDense::MultiplyMPI(...)" << endl;
#endif

    clock_t start, end;
    double compute_time = 0;
    double communicate_time = 0;
    double *all_time = 0;
    char **all_processor_name = 0;
    int *all_processor_namelen = 0;
    
    int node, total_node, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Status status;

    MPI_Comm_size( comm, &total_node );
    MPI_Comm_rank( comm, &node );
    MPI_Get_processor_name( processor_name, &namelen );

    //cout << "MPI total_node = " << total_node << endl;
    
    start = clock();
    int nrow1 = MatrixBase::GetNRow();
    int ncol1 = MatrixBase::GetNCol();
    int nrow2 = B.GetNRow();
    int ncol2 = B.GetNCol();
    end = clock();
    compute_time += (double)(end - start);

    assert( nrow1!=0 && ncol1!=0 && ncol2!=0 && ncol1==nrow2 );
    assert( nrow1%total_node==0 && ncol1%total_node==0 );// && ncol2%total_node==0 );

    start = clock();
    MatrixDense<Type> B_T = B.Transform();

    MatrixDense<Type> Ci;
    
    // 把 A 按行分成 total_node 份 Ai
    // 每个 node=i 计算 Ai*B
    // 把 B 转置，调用下面的函数
    // 未测试 直接做矩阵相乘 与 先转置再调用转置版本相乘 的效率
    int sub_nrow = nrow1/total_node;
    clock_t start11, end11;
    start11 = clock();
    Ci = MultiplySliceTransform( node*sub_nrow, (node+1)*sub_nrow-1,
	                         0, ncol1-1,
				 B_T,
				 0, ncol2-1, 0, nrow2-1 );
    end11 = clock();
    if(node == -1)
        printf("node = %d, time0 = %f\n", node, (double)(end11-start11)/CLOCKS_PER_SEC);
    MatrixDense<Type> C;
    if( node == 0 )
    {
	C = MatrixDense<Type>( nrow1, ncol2, "PRODUCT_MPI" );
	for( int i(0); i<sub_nrow; i++ )
	{
	    for( int j(0); j<ncol2; j++ )
	    {
		C.val[i][j] = Ci.val[i][j];
	    }
	}
    }
    
    MPI_Datatype datatype;
    if( sizeof(Type)==sizeof(int) )
	datatype = MPI_INT;
    else
	datatype = MPI_DOUBLE;

    Type *Ci_mpi = new Type [ncol2*sub_nrow];

    end = clock();
    compute_time += (double)(end - start);
    
    if( node != 0 )
    {
	start = clock();
	int k = 0;
	for( int i(0); i<sub_nrow; i++ )
	{
	    for( int j(0); j<ncol2; j++ )
	    {
		Ci_mpi[k++] = Ci.val[i][j];
	    }
	}
	
	//end = clock();
        //compute_time += (double)(end - start);
        //communicate_time += (double)(end - start);
        
        //start = clock();
        
	MPI_Send( Ci_mpi, ncol2*sub_nrow, datatype, 0, 1, comm );
	
	end = clock();
        communicate_time += (double)(end - start);
    }
    else
    {
	for( int n(1); n<total_node; n++ )
	{
	    start = clock();
	    
	    MPI_Recv( Ci_mpi, ncol2*sub_nrow, datatype, n, 1, comm, &status );
	    
	    //end = clock();
	    //printf("Recv from %d: time = %f\n", n, (double)(end - start)/CLOCKS_PER_SEC);
            //communicate_time += (double)(end - start);
            
            //start = clock();
	    int k = 0;
	    for( int i(0); i<sub_nrow; i++ )
	    {
		for( int j(0); j<ncol2; j++ )
		{
		    C.val[n*sub_nrow+i][j] = Ci_mpi[k++];
		}
	    }
	    end = clock();
            //compute_time += (double)(end - start);
            communicate_time += (double)(end - start);
            //printf("Recv from %d: time = %f\n", n, (double)(end - start)/CLOCKS_PER_SEC);
	    
	}
    }
    
    
    if( node != 0 )
    {
        compute_time /= CLOCKS_PER_SEC;
        communicate_time /= CLOCKS_PER_SEC;
	MPI_Send(     &compute_time,       1, MPI_DOUBLE, 0, 10, comm );
	MPI_Send( &communicate_time,       1, MPI_DOUBLE, 0, 20, comm );
	MPI_Send(          &namelen,       1,    MPI_INT, 0, 30, comm );
	MPI_Send(    processor_name, namelen+1,   MPI_CHAR, 0, 40, comm );// namelen+1 是为了最后一个字符 \0 也发送
    }
    else
    {
        compute_time /= CLOCKS_PER_SEC;
        communicate_time /= CLOCKS_PER_SEC;
        all_time = new double [total_node*2];
        all_time[0] = compute_time;
        all_time[1] = communicate_time;
        
	all_processor_name = new char* [total_node];
	all_processor_name[0] = new char [MPI_MAX_PROCESSOR_NAME];
	strcpy( all_processor_name[0], processor_name );
	
	all_processor_namelen = new int [total_node];
	all_processor_namelen[0] = namelen;
	
	for( int n(1); n<total_node; n++ )
	{
	    MPI_Recv( &all_time[2*n]  , 1, MPI_DOUBLE, n, 10, comm, &status );
	    MPI_Recv( &all_time[2*n+1], 1, MPI_DOUBLE, n, 20, comm, &status );
	    MPI_Recv( &all_processor_namelen[n], 1, MPI_INT, n, 30, comm, &status );
	    all_processor_name[n] = new char [MPI_MAX_PROCESSOR_NAME];
	    MPI_Recv( all_processor_name[n], all_processor_namelen[n]+1, MPI_CHAR, n, 40, comm, &status );
	}
	
	compute_time = 0;
	communicate_time = 0;
	printf("\n====================== mpi details ======================\n");
	printf("node       name     compute    communicate    total\n");
	for( int n(0); n<total_node; n++ )
	{
	    compute_time += all_time[2*n];
	    communicate_time += all_time[2*n+1];
	    printf("%3d %10s %12.6f %12.6f %12.6f\n", n, all_processor_name[n], 
	                all_time[2*n], all_time[2*n+1], all_time[2*n]+all_time[2*n+1] );
	    
	}
	printf("---------------------------------------------------------\n");
	printf("%3s %10s %12.6f %12.6f %12.6f\n", "-", "total", 
	                compute_time, communicate_time, compute_time+communicate_time);
	printf("=========================================================\n");
	delete [] all_time;
	delete [] all_processor_namelen;
	for( int n(0); n<total_node; n++ )
	    delete [] all_processor_name[n];
	delete [] all_processor_name;
    }
    

    delete [] Ci_mpi;

    return C;
}

template <typename Type>
int MatrixDense<Type>::IsZero(const Type tol) const
{
    int nrow = MatrixBase::GetNRow();
    int ncol = MatrixBase::GetNCol();
    Type max = (Type)0.0;

    for( int i(0); i<nrow; i++ )
    {
	for( int j(0); j<ncol; j++ )
	{
	    if( abs(val[i][j]) > tol )
	    {
		//printf("%20.10f,%20.10f\n", val[i][j],tol);
		return 0;
	    }
	}
    }

    return 1;
}

}
#endif
