#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float dpunroll(long N, float *pA, float *pB);

float dpunroll(long N, float *pA, float *pB)
{
    float R = 0.0;
    int j;
    for (j = 0; j < N; j += 16)
        R += pA[j] * pB[j] + pA[j + 1] * pB[j + 1] + pA[j + 2] * pB[j + 2] + pA[j + 3] * pB[j + 3] + pA[j + 4] * pB[j + 4] + pA[j + 5] * pB[j + 5] + pA[j + 6] * pB[j + 6] + pA[j + 7] * pB[j + 7] +  
                 pA[j + 8] * pB[j + 8] + pA[j + 9] * pB[j + 9] + pA[j + 10] * pB[j + 10] + pA[j + 11] * pB[j + 11] + pA[j + 12] * pB[j + 12] + pA[j + 13] * pB[j + 13] + pA[j + 14] * pB[j + 14] + pA[j + 15] * pB[j + 15];
    return R;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Usage: %s <N> <repetitions>\n", argv[0]);
        return 1;
    }

    long N = atol(argv[1]);
    int reps = atoi(argv[2]);

    // Init vectors
    float *pA = (float *)malloc(N * sizeof(float));
    float *pB = (float *)malloc(N * sizeof(float));
    for (long i = 0; i < N; i++)
    {
        pA[i] = 1.0f;
        pB[i] = 1.0f;
    }

    struct timespec start, end;
    double total_time = 0;

    for (int r = 0; r < reps; r++)
    {
        clock_gettime(CLOCK_MONOTONIC, &start);

        float result = dpunroll(N, pA, pB);

        clock_gettime(CLOCK_MONOTONIC, &end);

        if (r >= reps / 2)
        {
            double time_spent = (end.tv_sec - start.tv_sec) +
                                (end.tv_nsec - start.tv_nsec) / 1e9;
            total_time += time_spent;
        }
        // print time for each repetition
        printf("Repetition %d: Result = %.2f, Time = %.6f sec\n", r + 1, result, (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
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