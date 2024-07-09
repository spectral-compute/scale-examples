#include <vector>
#include <iostream>

#include <cublas_v2.h>


void check(cudaError_t error, const char * file, size_t line) {
    if (error != cudaSuccess)
    {
        std::cout << "cuda error: " << cudaGetErrorString(error) << " at " << file << ":" << line << std::endl;
        exit(1);
    }
}


void checkCublas(cublasStatus_t error, const char * file, size_t line) {
    if (error != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublas error: " << cublasGetStatusString(error) << " at " << file << ":" << line << std::endl;
        exit(1);
    }
}


#define CHECK(error) check(error, __FILE__, __LINE__)
#define CHECK_CUBLAS(error) checkCublas(error, __FILE__, __LINE__)


int main(int argc, char ** argv) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const size_t N = 10;
    const size_t BYTES = N * sizeof(double);
    const double E = 1e-5;

    /* Prepare the data */

    std::vector<double> A(N);
    std::vector<double> B(N);

    for (size_t i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i + N;
    }

    /* Send the data */

    double * devA;
    double * devB;

    CHECK(cudaMalloc(&devA, BYTES));
    CHECK(cudaMalloc(&devB, BYTES));

    CHECK(cudaMemcpy(devA, A.data(), BYTES, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(devB, B.data(), BYTES, cudaMemcpyHostToDevice));

    /* Calculate */

    const int strideA = 1;
    const int strideB = 1;
    double result = 0;

    CHECK_CUBLAS(cublasDdot(handle, A.size(), devA, strideA, devB, strideB, &result));

    CHECK(cudaDeviceSynchronize());

    double expected = 0;
    for (size_t i = 0; i < N; i++) {
        expected += A[i] * B[i];
    }

    if (std::abs(result - expected) > E) {
        std::cout << "Result " << result << " is different from expected " << expected << std::endl;
    }

    CHECK_CUBLAS(cublasDestroy(handle));

    std::cout << "Example finished." << std::endl;

    return 0;
}
