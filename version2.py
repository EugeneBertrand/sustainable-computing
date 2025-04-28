import multiprocessing as mp
import numpy as np
import time
import matplotlib.pyplot as plt
import psutil
from datetime import datetime

LOG_FILE_1 = "small_matrix_log_1.txt"
LOG_FILE_2 = "small_matrix_log_2.txt"

# Resource Estimator Based on Matrix Size (Adjusted)
def estimate_resources(matrix_size):
    # Estimate time more realistically: time proportional to N^2.8
    est_time = (matrix_size ** 2.8) * 1e-9  # Use empirical scaling
    est_cpu = 100.0  # Assume full core usage per operation
    return est_cpu, est_time

# Matrix Multiplication Task with Detailed Logging (from first code)
def small_matrix_multiply(run_id, matrix_size, iterations, interval):
    actual_durations = []
    estimated_times = []

    for i in range(iterations):
        A = np.random.rand(matrix_size, matrix_size)
        B = np.random.rand(matrix_size, matrix_size)

        est_cpu, est_time = estimate_resources(matrix_size)
        estimated_times.append(est_time)

        with open(LOG_FILE_1, "a") as f:
            f.write(f"{datetime.now()} | Run {run_id}-{i} | EST_CPU: {est_cpu:.2f}% | EST_TIME: {est_time:.4f}s\n")

        start_time = time.time()
        np.dot(A, B)
        duration = time.time() - start_time
        actual_durations.append(duration)
        cpu_percent = psutil.cpu_percent(interval=0.1)

        with open(LOG_FILE_1, "a") as f:
            f.write(f"{datetime.now()} | Run {run_id}-{i} | ACTUAL_DURATION: {duration:.4f}s | ACTUAL_CPU: {cpu_percent:.2f}%\n")

        time.sleep(interval)

# Matrix Multiplication Task without Detailed Logging (for parallelism)
def matrix_multiply(matrix_size, iterations):
    durations = []
    for _ in range(iterations):
        A = np.random.rand(matrix_size, matrix_size)
        B = np.random.rand(matrix_size, matrix_size)
        start_time = time.time()
        np.dot(A, B)
        durations.append(time.time() - start_time)
    return np.mean(durations)

# Measure Speedup from Parallelism with per-core entry and runtime chart
def measure_speedup(matrix_size, iterations):
    num_cores = mp.cpu_count()

    # Single-core
    single_start = time.time()
    single_duration = sum(matrix_multiply(matrix_size, iterations) for _ in range(num_cores))
    single_total_duration = time.time() - single_start

    # Multi-core
    multi_start = time.time()
    with mp.Pool(num_cores) as pool:
        core_durations = pool.starmap(matrix_multiply, [(matrix_size, iterations) for _ in range(num_cores)])
    multi_total_duration = time.time() - multi_start

    # Calculate Speedup
    total_speedup = single_total_duration / multi_total_duration
    per_core_speedups = [single_duration / (duration * num_cores) for duration in core_durations]

    with open(LOG_FILE_2, "a") as f:
        f.write(f"{datetime.now()} | Single-core Total Duration: {single_total_duration:.4f}s\n")
        f.write(f"{datetime.now()} | Multi-core Total Duration: {multi_total_duration:.4f}s\n")
        f.write(f"{datetime.now()} | Total Speedup: {total_speedup:.2f}x\n")
        for idx, speedup in enumerate(per_core_speedups):
            f.write(f"{datetime.now()} | Core-{idx} Speedup: {speedup:.2f}x\n")

    # Plotting Runtime Comparison
    plt.figure(figsize=(8, 6))
    plt.bar(['Single-Core Total', 'Multi-Core Total', 'Multi-Core Per-Core Avg'],
            [single_total_duration, multi_total_duration, multi_total_duration / num_cores],
            color=['blue', 'green', 'orange'])
    plt.ylabel('Runtime (s)')
    plt.title('Enhanced Matrix Multiplication Parallel Speedup')
    plt.tight_layout()
    plt.savefig('enhanced_parallelism_runtime.png')
    plt.show()

# CPU Behavior Monitor (Slope Detection)
def behavior_change_monitor(interval=1, threshold=10):
    prev_cpu = psutil.cpu_percent(interval=1)
    while True:
        time.sleep(interval)
        curr_cpu = psutil.cpu_percent(interval=1)
        delta = abs(curr_cpu - prev_cpu)

        if delta > threshold:
            with open(LOG_FILE_1, "a") as f:
                f.write(f"{datetime.now()} | Behavior Change Detected | CPU Δ: {delta:.2f}%\n")

        prev_cpu = curr_cpu

# Main Execution
if __name__ == "__main__":
    open(LOG_FILE_1, "w").close()
    open(LOG_FILE_2, "w").close()

    monitor = mp.Process(target=behavior_change_monitor)
    monitor.start()

    detailed_logging = mp.Process(target=small_matrix_multiply, args=("Detailed", 100, 20, 0))
    detailed_logging.start()
    detailed_logging.join()

    measure_speedup(matrix_size=500, iterations=10)

    monitor.terminate()

    print(f"Computation complete. Check logs at {LOG_FILE_1} and {LOG_FILE_2}")
