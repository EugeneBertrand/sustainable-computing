#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>

#define LOG_FILE_2 "small_matrix_log_2.txt"

const int matrix_sizes[] = {500, 1000, 1500};
const int num_sizes = sizeof(matrix_sizes) / sizeof(matrix_sizes[0]);

double rand_double() {
    return (double)rand() / RAND_MAX;
}

void matrix_multiply(double *A, double *B, double *C, int size) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double sum = 0.0;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

int main() {
    FILE *log2 = fopen(LOG_FILE_2, "w");

    printf("Detected %d threads (OpenMP)\n", omp_get_max_threads());

    for (int s = 0; s < num_sizes; s++) {
        int size = matrix_sizes[s];
        printf("Testing matrix size: %d x %d\n", size, size);
        fprintf(log2, "\n--- Matrix Size: %d x %d ---\n", size, size);

        double *A = malloc(size * size * sizeof(double));
        double *B = malloc(size * size * sizeof(double));
        double *C = malloc(size * size * sizeof(double));
        double *C_seq = malloc(size * size * sizeof(double));

        for (int i = 0; i < size * size; i++) {
            A[i] = rand_double();
            B[i] = rand_double();
        }

        // SINGLE-CORE (Sequential) — Disable OpenMP
        omp_set_num_threads(1);
        double start_single = omp_get_wtime();
        matrix_multiply(A, B, C_seq, size);
        double end_single = omp_get_wtime();
        double single_total_time = end_single - start_single;

        // MULTI-CORE (Parallel) — Restore OpenMP threads
        omp_set_num_threads(omp_get_max_threads());
        double start_multi = omp_get_wtime();
        matrix_multiply(A, B, C, size);
        double end_multi = omp_get_wtime();
        double multi_total_time = end_multi - start_multi;

        double speedup = single_total_time / multi_total_time;

        fprintf(log2, "%ld | Single-core Total Time: %.4fs\n", time(NULL), single_total_time);
        fprintf(log2, "%ld | Multi-core Total Time: %.4fs\n", time(NULL), multi_total_time);
        fprintf(log2, "%ld | Overall Speedup: %.2fx\n", time(NULL), speedup);

        free(A);
        free(B);
        free(C);
        free(C_seq);
    }

    fclose(log2);

    printf("Benchmarking complete. Check logs at %s\n", LOG_FILE_2);

    return 0;
}
