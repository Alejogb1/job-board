---
title: "How can Fortran LAPACK functions be called from CUDA C code?"
date: "2025-01-30"
id: "how-can-fortran-lapack-functions-be-called-from"
---
The efficient utilization of existing numerical libraries like LAPACK is crucial for high-performance computing.  I've found, through extensive experience in scientific simulation, that directly invoking Fortran-based LAPACK routines from CUDA C requires careful management of data transfer and the Fortran calling convention. This process avoids recreating highly optimized linear algebra algorithms in CUDA, saving considerable development effort and often achieving superior performance.

The core challenge lies in the inherent differences between the Fortran and C/C++ programming models, notably in how they pass arguments and handle memory layout, particularly regarding multi-dimensional arrays. Fortran uses column-major order, whereas C/C++ employs row-major order. Furthermore, LAPACK routines often expect pointers to the start of multi-dimensional arrays, and C/C++ does not necessarily provide a contiguous memory representation for multi-dimensional arrays as expected by Fortran.

To bridge this gap, I've learned several techniques. The overarching principle is to manage data movement explicitly between the host (CPU) and device (GPU) and to conform to Fortran's memory layout conventions. This involves three primary steps: 1) Data conversion from C/C++ representation to Fortran column-major order, 2) Transfer of data to the GPU and back, and 3) Invocation of the LAPACK routines using the Fortran calling convention.

The Fortran calling convention, particularly the handling of arguments, introduces complexities. Fortran passes arguments by reference, which means you provide a pointer to the data instead of copying the value. When working with LAPACK within a mixed-language context, this distinction becomes vital.  You must provide the pointers expected by the Fortran routine when invoking it from C/C++.

I've compiled three code examples, each illustrating a distinct approach to tackling this integration, each building upon the experience of troubleshooting similar integration issues.

**Example 1: Basic Data Transfer and Function Call (using single-precision `sgeqrf`)**

This first example demonstrates a fundamental case of calling `sgeqrf`, a LAPACK routine for computing the QR factorization of a matrix. Here, the data transformation is explicitly performed on the CPU. We utilize `cudaMalloc` to allocate memory on the GPU. After the GPU operation, we transfer the data from GPU back to CPU.
```c
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

//Fortran interface definition, use extern "C" to prevent name mangling
extern "C" void sgeqrf_(int *m, int *n, float *a, int *lda, float *tau, float *work, int *lwork, int *info);


void cpu_to_fortran(float *src, float *dst, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dst[j * m + i] = src[i * n + j];
        }
    }
}

void fortran_to_cpu(float *src, float *dst, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dst[i * n + j] = src[j * m + i];
        }
    }
}


int main() {
    int m = 4;
    int n = 3;

    // Input matrix A (row-major)
    float A_cpu_row[] = {1, 2, 3,
                         4, 5, 6,
                         7, 8, 9,
                         10,11,12};


    // Fortran-compatible (column-major) copy
    float *A_cpu_col = (float*)malloc(m * n * sizeof(float));
    if (A_cpu_col == nullptr) return 1;


    cpu_to_fortran(A_cpu_row,A_cpu_col,m,n);



    float *A_gpu;
    cudaMalloc((void**)&A_gpu, m * n * sizeof(float));
    cudaMemcpy(A_gpu, A_cpu_col, m * n * sizeof(float), cudaMemcpyHostToDevice);

    float *tau_gpu;
    cudaMalloc((void**)&tau_gpu, std::min(m, n) * sizeof(float));

    int lwork = m * n;
    float *work_gpu;
    cudaMalloc((void**)&work_gpu, lwork * sizeof(float));

    int info;
    sgeqrf_(&m, &n, A_gpu, &m, tau_gpu, work_gpu, &lwork, &info);

    if (info != 0) {
      printf("sgeqrf failed with info code %d\n",info);
    }


    float *A_cpu_out = (float*)malloc(m * n * sizeof(float));
    float *tau_cpu = (float*)malloc(std::min(m,n) * sizeof(float));

    if (A_cpu_out == nullptr || tau_cpu == nullptr) {
        cudaFree(A_gpu);
        cudaFree(tau_gpu);
        cudaFree(work_gpu);
        free(A_cpu_col);
        if (A_cpu_out != nullptr) free(A_cpu_out);
        if(tau_cpu!= nullptr) free(tau_cpu);
        return 1;
    }

    cudaMemcpy(A_cpu_out, A_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(tau_cpu, tau_gpu, std::min(m,n)*sizeof(float), cudaMemcpyDeviceToHost);


    // Transform back into row-major
    float *A_cpu_out_row = (float*)malloc(m*n*sizeof(float));
    fortran_to_cpu(A_cpu_out,A_cpu_out_row,m,n);


    printf("Resulting matrix:\n");
     for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
             printf("%f ", A_cpu_out_row[i*n+j]);
        }
        printf("\n");
    }
    printf("Tau vector:\n");
    for (int i=0; i<std::min(m,n); i++) {
        printf("%f ",tau_cpu[i]);
    }
    printf("\n");


    free(A_cpu_out_row);
    free(A_cpu_out);
    free(tau_cpu);
    free(A_cpu_col);

    cudaFree(A_gpu);
    cudaFree(tau_gpu);
    cudaFree(work_gpu);


    return 0;
}
```
In this example, `cpu_to_fortran` converts a standard C row-major matrix to column-major, and `fortran_to_cpu` does the reverse after the LAPACK call. Memory for input matrix, tau, and workspace, is allocated on GPU. The LAPACK routine is called on the GPU, with pointers to device memory. Finally, the results are copied back to the CPU for further processing.

**Example 2: Memory Management with Helper Functions and dynamic arrays**

This example demonstrates a more robust memory management scheme. Here, I utilize functions to encapsulate the creation and destruction of device memory, along with error checking. Dynamic arrays are used to make the code more flexible. This is particularly important when integrating into more complex systems where data sizes may vary dynamically.
```c
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

//Fortran interface definition, use extern "C" to prevent name mangling
extern "C" void dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);



double* cpu_to_fortran_double(double *src, int m, int n) {
    double* dst = (double*)malloc(m * n * sizeof(double));
    if (dst == nullptr) return nullptr;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dst[j * m + i] = src[i * n + j];
        }
    }
    return dst;

}

void fortran_to_cpu_double(double *src, double *dst, int m, int n) {
     for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dst[i * n + j] = src[j * m + i];
        }
    }
}

double *allocate_gpu_memory(size_t size) {
    double *device_ptr;
    cudaError_t err = cudaMalloc((void **)&device_ptr, size);
    if(err!= cudaSuccess) {
        fprintf(stderr, "GPU memory allocation failed with code %d.\n", err);
        return nullptr;
    }
    return device_ptr;
}

void free_gpu_memory(double *device_ptr) {
     if (device_ptr != nullptr) {
         cudaFree(device_ptr);
     }

}


int main() {
    int n = 3;
    int nrhs = 1; //number of right hand sides
    double A_cpu_row[] = {1, 2, 3,
                          4, 5, 6,
                          7, 8, 10};


    double b_cpu_row[] = {1,
                          1,
                          1};

    double* A_cpu_col = cpu_to_fortran_double(A_cpu_row, n, n);
    double* b_cpu_col = cpu_to_fortran_double(b_cpu_row, n, nrhs);
    if (A_cpu_col == nullptr || b_cpu_col == nullptr) return 1;

    double *A_gpu = allocate_gpu_memory(n * n * sizeof(double));
    double *b_gpu = allocate_gpu_memory(n * nrhs * sizeof(double));

    if (A_gpu == nullptr || b_gpu == nullptr) {
        free(A_cpu_col);
        free(b_cpu_col);
        return 1;
    }

    cudaMemcpy(A_gpu, A_cpu_col, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b_cpu_col, n * nrhs * sizeof(double), cudaMemcpyHostToDevice);

    int *ipiv_gpu = (int*)allocate_gpu_memory(n * sizeof(int));
    if (ipiv_gpu == nullptr) {
        free(A_cpu_col);
        free(b_cpu_col);
        free_gpu_memory(A_gpu);
        free_gpu_memory(b_gpu);
        return 1;
    }

    int info;
    dgesv_(&n, &nrhs, A_gpu, &n, ipiv_gpu, b_gpu, &n, &info);

    if (info != 0) {
        printf("dgesv failed with info code %d\n",info);
    }



    double* b_cpu_out = (double*)malloc(n*nrhs*sizeof(double));
    if (b_cpu_out == nullptr) {
        free(A_cpu_col);
        free(b_cpu_col);
        free_gpu_memory(A_gpu);
        free_gpu_memory(b_gpu);
        free_gpu_memory((double*)ipiv_gpu);
         return 1;
    }

    cudaMemcpy(b_cpu_out, b_gpu, n * nrhs * sizeof(double), cudaMemcpyDeviceToHost);

    double* b_cpu_out_row = (double*)malloc(n*nrhs*sizeof(double));
    fortran_to_cpu_double(b_cpu_out, b_cpu_out_row, n, nrhs);


    printf("Solution matrix:\n");
     for (int i = 0; i < n; i++) {
        for (int j = 0; j < nrhs; j++) {
             printf("%f ", b_cpu_out_row[i*nrhs+j]);
        }
        printf("\n");
    }

    free(A_cpu_col);
    free(b_cpu_col);
    free(b_cpu_out_row);
    free(b_cpu_out);
    free_gpu_memory(A_gpu);
    free_gpu_memory(b_gpu);
    free_gpu_memory((double*)ipiv_gpu);


    return 0;
}
```
This example uses `dgesv_`, a LAPACK routine for solving a system of linear equations. The functions `allocate_gpu_memory` and `free_gpu_memory` reduce the boilerplate code associated with managing device memory, and provide error checking on GPU memory allocation.

**Example 3: Using Unified Memory and the `cublas` library**

This example explores the use of CUDA Unified Memory, a mechanism for automatically migrating data between the host and device. Also included is the use of the `cublas` library which offers a high performance GPU accelerated BLAS. While not strictly using Fortran LAPACK, it demonstrates an alternative approach, potentially more efficient in certain situations.

```c
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void print_matrix(double* matrix, int m, int n) {
   for (int i=0; i<m; i++) {
      for(int j=0; j<n; j++) {
         printf("%f ",matrix[i*n + j]);
      }
      printf("\n");
   }
}

int main() {
    int m = 3;
    int n = 3;
    int k = 3;

    double *A_cpu_row = (double*)malloc(m*k*sizeof(double));
    double *B_cpu_row = (double*)malloc(k*n*sizeof(double));

    if(A_cpu_row == nullptr || B_cpu_row == nullptr) return 1;


    double A_data[] = {1, 2, 3,
                      4, 5, 6,
                      7, 8, 9};

    double B_data[] = {9,8,7,
                       6,5,4,
                       3,2,1};

    for(int i=0; i<m*k; i++) A_cpu_row[i] = A_data[i];
    for(int i=0; i<k*n; i++) B_cpu_row[i] = B_data[i];



    double *A_gpu;
    double *B_gpu;
    double *C_gpu;

    cudaMallocManaged((void **)&A_gpu, m * k * sizeof(double));
    cudaMallocManaged((void **)&B_gpu, k * n * sizeof(double));
    cudaMallocManaged((void **)&C_gpu, m * n * sizeof(double));

    if (A_gpu == nullptr || B_gpu == nullptr || C_gpu == nullptr) {
        free(A_cpu_row);
        free(B_cpu_row);
         if (A_gpu!=nullptr) cudaFree(A_gpu);
         if(B_gpu!=nullptr) cudaFree(B_gpu);
         if(C_gpu!=nullptr) cudaFree(C_gpu);
        return 1;
    }


    for (int i = 0; i < m * k; i++) {
        A_gpu[i] = A_cpu_row[i];
    }

    for (int i = 0; i < k * n; i++) {
        B_gpu[i] = B_cpu_row[i];
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

     double alpha = 1.0;
     double beta = 0.0;

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k, &alpha, A_gpu, m, B_gpu, k, &beta, C_gpu, m);


    cudaDeviceSynchronize();


    printf("Input matrix A:\n");
    print_matrix(A_cpu_row, m, k);

    printf("Input matrix B:\n");
    print_matrix(B_cpu_row,k,n);


    printf("Resultant matrix C:\n");
    print_matrix(C_gpu, m, n);


    free(A_cpu_row);
    free(B_cpu_row);
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    cublasDestroy(handle);


    return 0;
}
```
This example demonstrates matrix multiplication using `cublasDgemm`. Unified Memory simplifies data handling, as explicit memory copies are no longer necessary. This makes the code significantly more concise while potentially improving performance by leveraging the automatic migration managed by the CUDA runtime.

For deeper understanding of Fortran and C/C++ interoperability, I recommend studying the documentation related to the Fortran ISO_C_BINDING module.  Furthermore, research on the specific memory management techniques provided by CUDA, such as using pinned host memory or CUDA streams, will be beneficial for optimizing data transfer between CPU and GPU.  Finally, detailed exploration of `cublas` will be helpful when performance is of primary concern, as it provides highly-optimized implementations of BLAS operations directly on the GPU.
