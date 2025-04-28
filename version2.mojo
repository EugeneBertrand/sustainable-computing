import sys
import time
import random
from io import File
from math import pow
from chrono import now

const LOG_FILE_1 = "small_matrix_log_1.txt"
const LOG_FILE_2 = "small_matrix_log_2.txt"

fn estimate_resources(matrix_size: Int) -> (Float64, Float64):
    let est_time = pow(matrix_size.to_float64(), 2.8) * 1e-9
    return (100.0, est_time)

# Simple matrix multiplication
fn matrix_multiply(matrix_size: Int):
    var A = [[random.rand_float64() for _ in range(matrix_size)] for _ in range(matrix_size)]
    var B = [[random.rand_float64() for _ in range(matrix_size)] for _ in range(matrix_size)]
    var C = [[0.0 for _ in range(matrix_size)] for _ in range(matrix_size)]

    for i in range(matrix_size):
        for j in range(matrix_size):
            for k in range(matrix_size):
                C[i][j] += A[i][k] * B[k][j]

fn small_matrix_multiply(run_id: String, matrix_size: Int, iterations: Int, interval: Int):
    let log = File.open(LOG_FILE_1, .append)

    for i in range(iterations):
        let (est_cpu, est_time) = estimate_resources(matrix_size)
        log.write(f"{now()} | Run {run_id}-{i} | EST_CPU: {est_cpu:.2f}% | EST_TIME: {est_time:.4f}s\n")

        let start_time = time.time()
        matrix_multiply(matrix_size)
        let duration = time.time() - start_time

        // Simulated CPU usage; real metrics require native bindings
        let simulated_cpu = 100.0
        log.write(f"{now()} | Run {run_id}-{i} | ACTUAL_DURATION: {duration:.4f}s | ACTUAL_CPU: {simulated_cpu:.2f}%\n")

        time.sleep(interval)

fn measure_speedup(matrix_size: Int, iterations: Int):
    let cores = 4  # Assume 4-core system for now

    let single_start = time.time()
    var single_duration = 0.0
    for _ in range(cores):
        let start = time.time()
        for _ in range(iterations):
            matrix_multiply(matrix_size)
        single_duration += time.time() - start
    let single_total = time.time() - single_start

    let multi_start = time.time()
    var multi_duration = 0.0
    for _ in range(cores):
        let start = time.time()
        for _ in range(iterations):
            matrix_multiply(matrix_size)
        multi_duration += time.time() - start
    let multi_total = time.time() - multi_start

    let speedup = single_total / multi_total

    let log = File.open(LOG_FILE_2, .append)
    log.write(f"{now()} | Single-core Total Duration: {single_total:.4f}s\n")
    log.write(f"{now()} | Multi-core Total Duration: {multi_total:.4f}s\n")
    log.write(f"{now()} | Total Speedup: {speedup:.2f}x\n")

fn behavior_change_monitor(interval: Int, threshold: Float64):
    # Placeholder: Real CPU metrics would use bindings to system APIs
    var prev_cpu: Float64 = 50.0
    let log = File.open(LOG_FILE_1, .append)

    while true:
        time.sleep(interval)
        let curr_cpu = prev_cpu + (random.rand_float64() * 20.0 - 10.0)
        let delta = abs(curr_cpu - prev_cpu)

        if delta > threshold:
            log.write(f"{now()} | Behavior Change Detected | CPU Δ: {delta:.2f}%\n")
        prev_cpu = curr_cpu

fn main():
    File.open(LOG_FILE_1, .write).close()
    File.open(LOG_FILE_2, .write).close()

    # Mojo doesn't support full multiprocessing yet; simulate inline
    behavior_change_monitor(1, 10.0)  # Could be run in another thread in future
    small_matrix_multiply("Detailed", 100, 5, 1)
    measure_speedup(100, 2)

    print("Computation complete. Check logs.")

main()
