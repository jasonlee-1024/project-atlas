#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float dp(long N, float *pA, float *pB);

float dp(long N, float *pA, float *pB) {
    float R = 0.0;
    for (long j = 0; j < N; j++)
        R += pA[j] * pB[j];
    return R;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <N> <repetitions>\n", argv[0]);
        return 1;
    }

    long N = atol(argv[1]);
    int reps = atoi(argv[2]);

    // Init vectors
    float *pA = (float *)malloc(N * sizeof(float));
    float *pB = (float *)malloc(N * sizeof(float));
    for (long i = 0; i < N; i++) {
        pA[i] = 1.0f;
        pB[i] = 1.0f;
    }

    struct timespec start, end;
    double total_time = 0;

    for (int r = 0; r < reps; r++) {
        clock_gettime(CLOCK_MONOTONIC, &start);

        float result = dp(N, pA, pB);

        clock_gettime(CLOCK_MONOTONIC, &end);

        if (r >= reps / 2) {
            double time_spent = (end.tv_sec - start.tv_sec) +
                               (end.tv_nsec - start.tv_nsec) / 1e9;
            total_time += time_spent;
        }
        //print time for each repetition
        printf("Repetition %d: Result = %.2f, Time = %.6f sec\n", r + 1, result, (end.tv_sec - start.tv_sec) +
                               (end.tv_nsec - start.tv_nsec) / 1e9);
    }

    free(pA);
    free(pB);

    // Calculate average time
    double average_time = (double)total_time / (reps / 2);

    // Calculate bandwidth
    double bandwidth = (2 * N * sizeof(float)) / (average_time * 1e9);

    // Calculate FLOPS
    double flops = (2 * N) / average_time;

    printf("N: %ld <T>: %.6f sec B: %.3f GB/sec F: %.3f FLOP/sec\n", N, average_time, bandwidth, flops);

    return 0;
}