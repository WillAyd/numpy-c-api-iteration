cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cd build
make

python -c "import npyiters; import numpy as np; npyiters.print_2d(np.ones((5, 5)))"
