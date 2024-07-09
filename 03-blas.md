# Example: BLAS

This example shows how a math wrapper library for BLAS can be used.

The example uses the `cublasDdot()` function to calculate the dot product of two vectors.

cuBLAS APIs are forwarded to use the relevant ROCm APIs.
Note that the example links to `cublas` in its [`CMakeLists.txt`](#cmakeliststxt-used).
Just like in other examples, this allows seamless transition of projects to SCALE without code modification.

## Example source code

```cpp
---8<--- "public/examples/src/blas/blas.cu"
```

## `CMakeLists.txt` used

```cmake
---8<--- "public/examples/src/blas/CMakeLists.txt"
```
