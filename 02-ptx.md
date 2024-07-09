# Example: PTX

This example covers the following features on top of what was shown in the [basic example](./01-basic.md):

- defining `__device__` functions
    - `ptx_add()`
    - `ptx_lop3()`
- using C++ templates with `__device__` and `__global__` functions
    - `ptx_lop3()`
    - `kernelLop3()`
- using inline PTX Assembly `asm(...);` blocks
    - `ptx_add()`
    - `ptx_lop3()`

Build and run the example by following the [general instructions](./index.md).

## Extra info

- [Using inline PTX Assembly](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)
- [PTX ISA reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

PTX instructions used:

- [`add`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-add)
- [`lop3`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3)

## Example source code

```cpp
---8<--- "public/examples/src/ptx/ptx.cu"
```

## `CMakeLists.txt` used

```cmake
---8<--- "public/examples/src/ptx/CMakeLists.txt"
```
