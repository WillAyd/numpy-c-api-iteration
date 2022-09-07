cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cd build
make

python -c "import npyiters; import numpy as np; npyiters.simple_loop(np.ones(5))"
