import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt

# Output files
LOG_FILE = "/scratch/core_scaling_log.txt"
PLOT_FILE = "/scratch/core_speedup_plot.png"

# Test config
MATRIX_SIZE = 58000  # Adjust up/down for memory pressure
CORE_COUNTS = [1, 3, 4, 8, 16, 32, 64]

def generate_matrices(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    return A, B

# ✅ Fixed shared memory worker
def parallel_worker_shared(args):
    shm_name_A, shm_name_B, shape_A, shape_B, dtype, row_start, row_end = args
    shm_A = shared_memory.SharedMemory(name=shm_name_A)
    shm_B = shared_memory.SharedMemory(name=shm_name_B)  # ✅ FIXED here

    A = np.ndarray(shape_A, dtype=dtype, buffer=shm_A.buf)
    B = np.ndarray(shape_B, dtype=dtype, buffer=shm_B.buf)
    result = np.dot(A[row_start:row_end], B)

    shm_A.close()
    shm_B.close()
    return result

def multi_core_matmul_shared(A, B, num_processes):
    n = A.shape[0]
    dtype = A.dtype

    shm_A = shared_memory.SharedMemory(create=True, size=A.nbytes)
    shm_B = shared_memory.SharedMemory(create=True, size=B.nbytes)

    A_sh = np.ndarray(A.shape, dtype=dtype, buffer=shm_A.buf)
    B_sh = np.ndarray(B.shape, dtype=dtype, buffer=shm_B.buf)
    np.copyto(A_sh, A)
    np.copyto(B_sh, B)

    chunk_size = n // num_processes
    tasks = []
    for i in range(num_processes):
        row_start = i * chunk_size
        row_end = (i + 1) * chunk_size if i != num_processes - 1 else n
        tasks.append((shm_A.name, shm_B.name, A.shape, B.shape, dtype, row_start, row_end))

    start = time.perf_counter()
    with mp.Pool(processes=num_processes) as pool:
        pool.map(parallel_worker_shared, tasks)
    elapsed = time.perf_counter() - start

    shm_A.close(); shm_A.unlink()
    shm_B.close(); shm_B.unlink()

    return elapsed

if __name__ == "__main__":
    mp.set_start_method("fork")
    A, B = generate_matrices(MATRIX_SIZE)

    core_counts = []
    runtimes_sec = []
    speedups = []

    with open(LOG_FILE, "w") as log:
        log.write(f"Matrix size: {MATRIX_SIZE} x {MATRIX_SIZE}\n")
        log.write(f"Started at: {datetime.now()}\n\n")

        baseline_time = None

        for cores in CORE_COUNTS:
            print(f"Running with {cores} cores...")
            try:
                elapsed_sec = multi_core_matmul_shared(A, B, cores)
                core_counts.append(cores)
                runtimes_sec.append(elapsed_sec)

                if cores == 1:
                    baseline_time = elapsed_sec
                    speedup = 1.0
                else:
                    speedup = baseline_time / elapsed_sec

                speedups.append(speedup)
                log.write(f"{cores} cores → {elapsed_sec:.2f} sec → Speedup: {speedup:.2f}x\n")
                print(f"Done {cores} cores: {elapsed_sec:.2f}s → Speedup: {speedup:.2f}x")
            except Exception as e:
                log.write(f"{cores} cores → ERROR: {e}\n")
                print(f"ERROR at {cores} cores: {e}")

    # Plot: Speedup vs Cores
    plt.figure()
    plt.plot(core_counts, speedups, marker='o')
    plt.xlabel("Number of Cores")
    plt.ylabel("Speedup (T₁ / Tₙ)")
    plt.title(f"Speedup vs Cores (Matrix: {MATRIX_SIZE}x{MATRIX_SIZE})")
    plt.grid(True)
    plt.savefig(PLOT_FILE)
    print(f"\n✅ Saved speedup plot to: {PLOT_FILE}")

