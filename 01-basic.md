# Example: Basic

This example covers the basic features of the CUDA language:

- defining a kernel (`__global__ basicSum()`)
- using CUDA APIs such as `cudaMalloc()`, `cudaFree()` and `cudaMemcpy()`
- launching a kernel

The example consists of:

- Generating test data on the host
- Sending data to the device
- Launching a kernel on the device
- Receiving data back from the device
- Checking that the data we received is what we expect

Build and run the example by following the [general instructions](./index.md).

## Example source code

```cpp
---8<--- "public/examples/src/basic/basic.cu"
```

## `CMakeLists.txt` used

```cmake
---8<--- "public/examples/src/basic/CMakeLists.txt"
```
