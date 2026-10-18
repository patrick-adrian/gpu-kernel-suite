CPU reference implementations live as inline functions in
`include/kernels/cpu_kernels.hpp` for simplicity and to keep the host-only
build free of a separate library dependency. This directory is reserved for
future out-of-line CPU sources (e.g., OpenMP parallel references) if needed.
