#!/bin/sh

set -e

SCALE_DIR="$1"
EXAMPLE="$2"

export PATH="${SCALE_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${SCALE_DIR}/lib:${LD_LIBRARY_PATH}"


case "${EXAMPLE}" in

    "basic" | "blas" | "ptx")
        rm -rf "src/${EXAMPLE}/build"

        cmake \
            -DCUDAToolkit_ROOT="${SCALE_DIR}" \
            -DCMAKE_CUDA_COMPILER="${SCALE_DIR}/bin/nvcc" \
            -DCMAKE_CUDA_ARCHITECTURES="86" \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -B "src/${EXAMPLE}/build" \
            "src/${EXAMPLE}"

        make \
            -C "src/${EXAMPLE}/build"

        export REDSCALE_EXCEPTIONS=1

        "src/${EXAMPLE}/build/example_${EXAMPLE}"
    ;;

    *)
        echo "Usage: $0 {PATH_TO_SCALE} {basic|blas|ptx}"
    ;;

esac
