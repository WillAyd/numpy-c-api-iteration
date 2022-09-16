# NumPy C API Iteration Examples

## Getting Started

Assuming you have CMake installed, you can create an out of source build folder to build and import the shared library from.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cd build
make
```

## Running the Example

```sh
python -c "import npyiters; import numpy as np; npyiters.print_2d(np.ones((3, 2)))"
```
