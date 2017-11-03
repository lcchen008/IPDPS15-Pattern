export OMP_NUM_THREADS=12
export MV2_ENABLE_AFFINITY=0
mpiexec -np 32 -pernode ./main
