// SpMV Code
// Created: 03-12-2019
// Author: Najeeb Ahmad
// Updated: 13-05-2020
// Author: Muhammad Aditya Sasongko
// Updated: 07-06-2020
// Author: Doruk Taneli

#include <bits/stdc++.h>
#include "common.h"
#include "matrix.h"
#include "mmio.h"

using namespace std;

int main(int argc, char **argv)
{
  csr_matrix matrix;
  string matrix_name;
  int P, myrank, M;
  double *myVecData, *myMatVal, *result, *rhs, *totalResult, *sendVecData;
  int *myColInd, *myRowptr;

  // Initializations
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  if (myrank = 0) //master
  {
    if (argc < 3)
    {
      std::cout << "Error: Missing arguments\n";
      std::cout << "Usage: " << argv[0] << " matrix.mtx\n"
                << " time_step_count";
      return EXIT_FAILURE;
    }

    printf("Reading .mtx file\n");
    int retCode = 0;
    int time_steps = atoi(argv[2]);
    matrix_name = argv[1];
    cout << matrix_name << endl;

    //double *rhs;
    //double *result;
    retCode = mm_read_unsymmetric_sparse(argv[1], &matrix.m, &matrix.n, &matrix.nnz,
                                         &matrix.csrVal, &matrix.csrRowPtr, &matrix.csrColIdx);

    if (retCode == -1)
    {
      cout << "Error reading input .mtx file\n";
      return EXIT_FAILURE;
    }

    printf("Matrix Rows: %d\n", matrix.m);
    printf("Matrix Cols: %d\n", matrix.n);
    printf("Matrix nnz: %d\n", matrix.nnz);
    coo2csr_in(matrix.m, matrix.nnz, matrix.csrVal, matrix.csrRowPtr, matrix.csrColIdx);
    printf("Done reading file\n");
  }

  // Scatter matrix entries to each processor
  // by sending partial Row pointers, Column Index and Values
  MPI_Scatter(eCount, 1, MPI_INT, &myNumE, 1, MPI_INT, 0, MPI_COMM_WORLD);

  myRowptr = (int *)malloc(sizeof(int) * (vSize + 1));
  MPI_Scatterv(rowptr, vecDataSize, vecDataDispls, MPI_INT,
               myRowptr, vSize, MPI_INT, 0, MPI_COMM_WORLD);
  myRowptr[vSize] = myNumE;

  myColInd = (int *)malloc(sizeof(int) * myNumE);
  MPI_Scatterv(colInd, eCount, eDispls, MPI_INT,
               myColInd, myNumE, MPI_INT, 0, MPI_COMM_WORLD);

  myMatVal = (double *)malloc(sizeof(double) * myNumE);
  MPI_Scatterv(matVal, eCount, eDispls, MPI_DOUBLE,
               myMatVal, myNumE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  free(rowptr);
  free(colInd);
  free(matVal);

  // Allocate vector rhs
  rhs = (double *)malloc(sizeof(double) * matrix.n);
  // Allocate vector result
  result = (double *)malloc(sizeof(double) * matrix.n);

  // Initialize right-hand-side
  for (int i = 0; i < matrix.n; i++)
    rhs[i] = (double)1.0 / matrix.n;

  clock_t start, end;

  if (myrank = 0) //master
    start = clock();

  for (int k = 0; k < time_steps; k++)
  {
    #pragma omp parallel for shared(result) schedule(dynamic)
    for (int i = 0; i < matrix.m; i++)
    {
      result[i] = 0.0;
      for (int j = matrix.csrRowPtr[i]; j < matrix.csrRowPtr[i + 1]; j++)
      {
        #pragma omp atomic update
        result[i] += matrix.csrVal[j] * rhs[matrix.csrColIdx[j]];
      }
    }

    //Gather and broadcast result in a more efficient way
    MPI_Allgather(result, M, MPI_DOUBLE, x, N, MPI_DOUBLE, MPI_COMM_WORLD);

    for (int i = 0; i < matrix.m; i++)
    {
      rhs[i] = result[i];
    }
  }

  if (myrank = 0) //master
  {
    end = clock();

    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    cout << "Time taken by program is : " << fixed
         << time_taken << setprecision(5);
    cout << " sec " << endl;
  }

  return EXIT_SUCCESS;
}
