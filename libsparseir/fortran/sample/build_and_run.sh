rm -rf build
cmake -B build -DSparseIR_DIR=$HOME/opt/libsparseir/share/cmake/SparseIR -DUSE_SYSTEM_LIBSPARSEIR=ON
cmake --build build

./build/second_order_perturbation_fort
