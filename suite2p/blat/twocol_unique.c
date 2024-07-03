/*
* A super hacky way to find unique rows in a two-columns array.
* Data needs to be presented in row-major order with type float64 (double).
*/

#include <stdlib.h>
#include <math.h>

int compare_c1 (const void *a, const void *b) {
    double *x = (double *) a;
    double *y = (double *) b;
    if (*x > *y)
        return 1;
    if (*x < *y)
        return -1;
    return 0;
}

int compare_c2 (const void *a, const void *b) {
    double *x = (double *) a + 1;
    double *y = (double *) b + 1;
    if (*x > *y)
        return 1;
    if (*x < *y)
        return -1;
    return 0;
}

void rankdata (double *x, unsigned int *ranks, size_t N) {
    unsigned int acc = 0;
    double pre = *x;
    for (int i = 0; i < N; i++) {
        if (pre != x[i*2]) {
            acc++;
            pre = x[i*2];
        }
        ranks[i*2] = acc;
    }
}

void twocol_unique (double *x, size_t N) {
/**** RADIX-inspired hack ****/
    unsigned int ranks[N * 2];

    qsort(x, N, sizeof(double) * 2, compare_c2); // assume little-endian
    rankdata(x + 1, ranks + 1, N); // rank second column

    int base = 0;
    int bucket = 1;
    while (bucket < N) {
        while (ranks[base*2 + 1] == ranks[bucket*2 + 1])
            bucket++;
        if ((bucket - base) > 1) {
            qsort(x + base*2, bucket - base, sizeof(double) * 2, compare_c1);
            base = bucket - 1;
        }
        base++;
        bucket++;
    }
    rankdata(x, ranks, N);

    unsigned long long *combined = (unsigned long long *) ranks;
    unsigned long long pre = *combined + 1;
    double *bp = x;
    for (int i = 0; i < N; i++) {
        if (pre != combined[i]) {
            *bp++ = x[i*2];
            *bp++ = x[i*2 + 1];
            pre = combined[i];
        }
    }

    while (bp < (x + N*2))
        *bp++ = NAN;
}

void kdtree (double *x, size_t N, double *node) {}
