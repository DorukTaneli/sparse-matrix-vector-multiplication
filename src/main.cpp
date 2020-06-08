// SpMV Code
// Created: 03-12-2019
// Author: Najeeb Ahmad
// Updated: 13-05-2020
// Author: Muhammad Aditya Sasongko
// Updated: 07-06-2020
// Author: Doruk Taneli

#include <bits/stdc++.h>
#include <mpi.h>
#include <omp.h>
#include "common.h"
#include "matrix.h"
#include "mmio.h"

using namespace std;

int main(int argc, char **argv)
{
  clock_t start, end;
  csr_matrix matrix;
  string matrix_name;
  int num_procs, myrank, M, tot_omp_threads, omp_threads_per_mpi, vSize, vSizeLast, myNumE;
  double *myVecData, *myMatVal, *result, *myResult, *rhs, *sendVecData;
  int time_steps, N, *myColInd, *myRowptr;

  // Initializations
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  int vecDataSize[num_procs], vecDataDispls[num_procs];
  int eCount[num_procs], eDispls[num_procs];

  tot_omp_threads = 16 / num_procs;
  omp_threads_per_mpi = tot_omp_threads / num_procs;

  if (myrank == 0) //master
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
    time_steps = atoi(argv[2]);
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

    N = matrix.n;
    M = N / num_procs; // Assuming N is a multiple of P

    vSize = ceil((double)matrix.n / num_procs);     //vector size for each process
    vSizeLast = matrix.n - (num_procs - 1) * vSize; //vector size for last process

    printf("starting vecData for loop\n");
    // Vector size and displacements for each processor
    for (int p = 0; p < num_procs - 1; p++)
    {
      vecDataSize[p] = vSize;
      vecDataDispls[p] = p * vSize;
    }
    vecDataSize[num_procs - 1] = vSizeLast;
    vecDataDispls[num_procs - 1] = (num_procs - 1) * vSize;

    printf("starting eCount for loop\n");
    // Matrix entries count and displacements for each processor
    for (int p = 0; p < num_procs; p++)
    {
      eCount[p] = matrix.csrRowPtr[p * vSize + vecDataSize[p]] - matrix.csrRowPtr[p * vSize];
      eDispls[p] = matrix.csrRowPtr[p * vSize];
    }
  }

  printf("MPI Scatter\n");
  // Scatter matrix entries to each processor
  // by sending partial Row pointers, Column Index and Values
  MPI_Scatter(eCount, 1, MPI_INT, &myNumE, 1, MPI_INT, 0, MPI_COMM_WORLD);

  printf("MPI Scatterv rowPtr\n");
  myRowptr = (int *)malloc(sizeof(int) * (vSize + 1));
  MPI_Scatterv(matrix.csrRowPtr, vecDataSize, vecDataDispls, MPI_INT,
               myRowptr, vSize, MPI_INT, 0, MPI_COMM_WORLD);
  myRowptr[vSize] = myNumE;
  printf("rowPtr scattered\n");

  myColInd = (int *)malloc(sizeof(int) * myNumE);
  MPI_Scatterv(matrix.csrColIdx, eCount, eDispls, MPI_INT,
               myColInd, myNumE, MPI_INT, 0, MPI_COMM_WORLD);
  printf("colInd scattered\n");

  myMatVal = (double *)malloc(sizeof(double) * myNumE);
  MPI_Scatterv(matrix.csrVal, eCount, eDispls, MPI_DOUBLE,
               myMatVal, myNumE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  printf("csrVal scattered\n");

  start = clock();

  // Allocate vector rhs
  rhs = (double *)malloc(sizeof(double) * matrix.n);
  // Allocate vector result
  result = (double *)malloc(sizeof(double) * matrix.n);
  // Allocate vector myresult
  myResult = (double *)malloc(sizeof(double) * vSize);

  // Initialize right-hand-side
  for (int i = 0; i < matrix.n; i++)
    rhs[i] = (double)1.0 / matrix.n;

  for (int k = 0; k < time_steps; k++)
  {
    #pragma omp parallel for shared(result) num_threads(omp_threads_per_mpi) schedule(dynamic)
    for (int i = 0; i < matrix.n; i++)
    {
      myResult[i] = 0.0;
      for (int j = myRowptr[i]; j < myRowptr[i + 1]; j++)
      {
        #pragma omp atomic update
        myResult[i] += myMatVal[j] * rhs[myColInd[j]];
      }
    }

    //Gather and broadcast result in a more efficient way
    MPI_Allgatherv(myResult, vSize, MPI_DOUBLE, result, vecDataSize, vecDataDispls, MPI_DOUBLE, MPI_COMM_WORLD);
    printf("Allgathered\n");

    for (int i = 0; i < matrix.m; i++)
    {
      rhs[i] = result[i];
    }
  }

  MPI_Finalize();

  end = clock();

  double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
  cout << "Time taken by program is : " << fixed
       << time_taken << setprecision(5);
  cout << " sec " << endl;

  return EXIT_SUCCESS;
}
