/*
* A super hacky way to find unique rows in a two-columns array.
* Data needs to be presented in row-major order with type float64 (double).
* Of course, it would have been probably even faster in column-major order.
* To that, I say, FUCK NUMPY!
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

struct sortgroup {
    unsigned int idx;
    unsigned long long comb;
    double *addr;
};

int compare_c1 (const void *a, const void *b) {
    double *x = (*(struct sortgroup*) a).addr;
    double *y = (*(struct sortgroup*) b).addr;
    if (*x > *y)
        return 1;
    if (*x < *y)
        return -1;
    return 0;
}

int compare_c2 (const void *a, const void *b) {
    double *x = (*(struct sortgroup*) a).addr + 1;
    double *y = (*(struct sortgroup*) b).addr + 1;
    if (*x > *y)
        return 1;
    if (*x < *y)
        return -1;
    return 0;
}

int compare (const void *a, const void *b) {
    unsigned long long x = (*(struct sortgroup*) a).comb;
    unsigned long long y = (*(struct sortgroup*) b).comb;
    if (x > y)
        return 1;
    if (x < y)
        return -1;
    return 0;
}

void rankdata (struct sortgroup *sorted, size_t N, int col) {
    unsigned int acc = 0;
    unsigned int *ptr;
    double pre = *(sorted[0].addr + col);
    for (int i = 0; i < N; i++) {
        if (pre != *(sorted[i].addr + col)) {
            acc++;
            pre = *(sorted[i].addr + col);
        }
        ptr = (unsigned int*) &sorted[i].comb;
        *(ptr + col) = acc;
    }
}

double *twocol_unique (double *x, size_t N) {
    struct sortgroup sorted[N];
    double *uniques = (double *) malloc(sizeof(double) * N * 2);

    for (int i = 0; i < N; i++) {
        sorted[i].idx = i;
        sorted[i].addr = x + i*2;
    }
    qsort(sorted, N, sizeof(*sorted), compare_c1);
    rankdata(sorted, N, 0);
    qsort(sorted, N, sizeof(*sorted), compare_c2);
    rankdata(sorted, N, 1);
    qsort(sorted, N, sizeof(*sorted), compare);

    unsigned long long pre = sorted[0].comb + 1;
    unsigned int incr = 0;
    for (int i = 0; i < N; i++) {
        if (pre != sorted[i].comb) {
            uniques[incr*2] = *sorted[i].addr;
            uniques[incr*2 + 1] = *(sorted[i].addr + 1);
            incr++;
            pre = sorted[i].comb;
        }
    }
    uniques[incr*2] = NAN;

    return uniques;
}

