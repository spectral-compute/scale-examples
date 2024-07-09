#include <bitset>
#include <vector>
#include <iostream>
#include <cstdint>


__device__ inline uint32_t ptx_add(uint32_t x, uint32_t y) {
    // Calculate a sum of `x` and `y`, put the result into `x`
    asm(
        "add.u32 %0, %0, %1;"
        : "+r"(x)
        : "r"(y)
    );
    return x;
}


__global__ void kernelAdd(const uint32_t * a, const uint32_t * b, size_t n, uint32_t * out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n)
    {
        out[idx] = ptx_add(a[idx], b[idx]);
    }
}


template<uint8_t Op>
__device__ inline uint32_t ptx_lop3(uint32_t x, uint32_t y, uint32_t z) {
    // Compute operator `Op` on `x`, `y`, `z`, put the result into `x`

    asm(
        "lop3.b32 %0, %0, %1, %2, %3;"
        : "+r"(x)
        : "r"(y), "r"(z), "n"(Op)
    );
    return x;
}


template<uint8_t Op>
__global__ void kernelLop3(const uint32_t * a, const uint32_t * b, const uint32_t * c, size_t n, uint32_t * out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n)
    {
        out[idx] = ptx_lop3<Op>(a[idx], b[idx], c[idx]);
    }
}


void check(cudaError_t error, const char * file, size_t line) {
    if (error != cudaSuccess)
    {
        std::cout << "cuda error: " << cudaGetErrorString(error) << " at " << file << ":" << line << std::endl;
        exit(1);
    }
}


#define CHECK(error) check(error, __FILE__, __LINE__)


template<typename T>
constexpr T lop3op(T a, T b, T c) {
    return a & b ^ (~c);
}


int main(int argc, char ** argv) {

    const size_t N = 4096;
    const size_t BYTES = N * sizeof(uint32_t);

    std::vector<uint32_t> a(N);
    std::vector<uint32_t> b(N);
    std::vector<uint32_t> c(N);
    std::vector<uint32_t> out(N);

    for (size_t i = 0; i < N; i++) {
        a[i] = i * 2;
        b[i] = N - i;
        c[i] = i * i;
    }

    uint32_t * devA;
    uint32_t * devB;
    uint32_t * devC;
    uint32_t * devOut;

    CHECK(cudaMalloc(&devA, BYTES));
    CHECK(cudaMalloc(&devB, BYTES));
    CHECK(cudaMalloc(&devC, BYTES));
    CHECK(cudaMalloc(&devOut, BYTES));

    CHECK(cudaMemcpy(devA, a.data(), BYTES, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(devB, b.data(), BYTES, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(devC, c.data(), BYTES, cudaMemcpyHostToDevice));

    // Test "add"

    kernelAdd<<<N / 256 + 1, 256>>>(devA, devB, N, devOut);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(out.data(), devOut, BYTES, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; i++) {
        if (a[i] + b[i] != out[i]) {
            std::cout << "Incorrect add: " << a[i] << " + " << b[i] << " = " << out[i] << " ?\n";
        }
    }

    // Test "lop3"

    constexpr uint8_t TA = 0xF0;
    constexpr uint8_t TB = 0xCC;
    constexpr uint8_t TC = 0xAA;
    constexpr uint8_t Op = lop3op(TA, TB, TC);

    kernelLop3<Op><<<N / 256 + 1, 256>>>(devA, devB, devC, N, devOut);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(out.data(), devOut, BYTES, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; i++) {
        if (lop3op(a[i], b[i], c[i]) != out[i]) {
            std::cout << "Incorrect lop3: \n"
                << "    " << std::bitset<32>{a[i]} << "\n"
                << " &  " << std::bitset<32>{b[i]} << "\n"
                << " ^ ~" << std::bitset<32>{c[i]} << "\n"
                << " =  " << std::bitset<32>{out[i]} << " ?\n\n";
        }
    }

    CHECK(cudaFree(devA));
    CHECK(cudaFree(devB));
    CHECK(cudaFree(devC));
    CHECK(cudaFree(devOut));

    // Finish

    std::cout << "Example finished" << std::endl;

    return 0;
}
