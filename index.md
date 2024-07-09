# SCALE by example

Whether you are starting a new project using SCALE from scratch, or adding it to an existing project, these SCALE usage examples can be useful for you.

The examples don't aim to cover all the things available with SCALE.
Instead, they highlight individual features in isolation from each other.
This way, they can be used as a reference in your development process.

Additionally, you are welcome to use these examples as a starting point for your project.

## List of examples

Here is the list of examples that are currently available.
Read more about them in their corresponding pages.

| Example                   | What it is about           |
| ------------------------- | -------------------------- |
| [Basic](./01-basic.md) | Usage in its simplest form |
| [PTX](./02-ptx.md)     | Using PTX Assembly         |
| [BLAS](./03-blas.md)   | Using BLAS maths wrapper   |

## Accessing the examples

The examples are hosted in a public repository.
You can clone it using git:

```sh
git clone https://github.com/spectral-compute/scale-examples.git
cd scale-examples
```

You can also download it as a ZIP file:

```sh
wget -O scale-examples.zip https://github.com/spectral-compute/scale-examples/archive/refs/heads/main.zip
unzip scale-examples.zip
cd scale-examples-main
```

## Using the examples

To build and run the examples, you should have SCALE [installed on your machine](../manual/01-installing.md).
You should also determine which [path to SCALE](../manual/02-how-to-use.md#identifying-gpu-target) to use, as it depends on your target GPU.

The example repository includes a helper script, `example.sh`, that configures, builds and runs the example of your choice.

Here is how you can use it for the [Basic](./01-basic.md) example:

```sh
# You should be in the root directory of the repository when running this
./example.sh {SCALE_DIR} basic
```

For the specified example, this will:

1. Remove its build directory if it already exists
2. Configure CMake for that example in a freshly-created build directory
3. Build the example in that directory using Make
4. Set the `REDSCALE_EXCEPTIONS=1` environment variable for better error reporting (read more [in the manual][exceptions])
4. Run the example

[exceptions]: ../manual/03-troubleshooting.md#exceptions

---

For accessibilty, SCALE documentation portal includes the source code of the examples in its pages.
This is the source code of `example.sh` referenced above:

```sh
---8<--- "public/examples/example.sh"
```
