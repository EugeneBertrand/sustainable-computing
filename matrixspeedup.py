import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import time
from datetime import datetime

# Output log file on Desktop
OUTPUT_FILE = os.path.join(os.path.expanduser("~"), "Desktop", "matrix_speedup_log.txt")

# Matrix sizes to test
matrix_sizes = [500, 2000, 5000, 10000]  # Avoid 20000+ unless you have massive RAM

def generate_matrices(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    return A, B

def single_core_matmul(A, B):
    start = time.perf_counter()
    C = np.dot(A, B)
    return time.perf_counter() - start

def parallel_worker_shared(args):
    shm_name_A, shm_name_B, shape_A, shape_B, dtype, row_start, row_end = args
    shm_A = shared_memory.SharedMemory(name=shm_name_A)
    shm_B = shared_memory.SharedMemory(name=shm_name_B)

    A = np.ndarray(shape_A, dtype=dtype, buffer=shm_A.buf)
    B = np.ndarray(shape_B, dtype=dtype, buffer=shm_B.buf)

    result = np.dot(A[row_start:row_end], B)

    shm_A.close()
    shm_B.close()

    return result

def multi_core_matmul_shared(A, B, num_processes):
    n = A.shape[0]
    dtype = A.dtype

    # Create shared memory
    shm_A = shared_memory.SharedMemory(create=True, size=A.nbytes)
    shm_B = shared_memory.SharedMemory(create=True, size=B.nbytes)

    A_sh = np.ndarray(A.shape, dtype=dtype, buffer=shm_A.buf)
    B_sh = np.ndarray(B.shape, dtype=dtype, buffer=shm_B.buf)

    np.copyto(A_sh, A)
    np.copyto(B_sh, B)

    # Chunking the rows
    chunk_size = n // num_processes
    tasks = []
    for i in range(num_processes):
        row_start = i * chunk_size
        row_end = (i + 1) * chunk_size if i != num_processes - 1 else n
        tasks.append((shm_A.name, shm_B.name, A.shape, B.shape, dtype, row_start, row_end))

    # Run processes
    start = time.perf_counter()
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(parallel_worker_shared, tasks)
    elapsed = time.perf_counter() - start

    # Combine and cleanup
    C = np.vstack(results)
    shm_A.close(); shm_A.unlink()
    shm_B.close(); shm_B.unlink()

    return elapsed, C

def log_results(matrix_size, t_single, t_multi, file):
    speedup = t_single / t_multi if t_multi > 0 else float('inf')
    with open(file, "a") as f:
        f.write(f"--- Matrix Size: {matrix_size} x {matrix_size} ---\n")
        f.write(f"Single-core Time: {t_single:.4f} sec\n")
        f.write(f"Multi-core Time:  {t_multi:.4f} sec\n")
        f.write(f"Speedup:          {speedup:.2f}x\n\n")

if __name__ == "__main__":
    mp.set_start_method("fork")  # Safer on Unix/macOS
    num_cores = mp.cpu_count()
    print(f"Detected {num_cores} cores")

    # Init log
    with open(OUTPUT_FILE, "w") as f:
        f.write(f"Matrix Multiplication Speedup Log\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Cores detected: {num_cores}\n\n")

    for size in matrix_sizes:
        print(f"Testing matrix size: {size} x {size}")
        A, B = generate_matrices(size)

        # Single-core
        t_single = single_core_matmul(A, B)

        # Multi-core
        t_multi, _ = multi_core_matmul_shared(A, B, num_cores)

        # Log
        log_results(size, t_single, t_multi, OUTPUT_FILE)
        print(f"Done size {size} → Speedup: {t_single / t_multi:.2f}x\n")

    print(f"Results saved to: {OUTPUT_FILE}")

