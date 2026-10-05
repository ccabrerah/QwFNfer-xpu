#pragma once
// Greedy argmax over a logit row, as `for (v) if (lg[v] > lg[best]) best = v` gives it: the first index of the
// largest value, NaNs never chosen (a NaN at index 0 is kept, since nothing compares greater). That loop carries
// `best` through every step and does not vectorize: 0.69 ms per 248K-entry row on the B70 host, two or more rows per
// verify step. Here: the maximum in 8 independent lanes (vectorizes), then the first index equal to it -- 0.03 ms.
#include <cstdint>

static inline int qwfn_argmax(const float * lg, int64_t n) {
    float m[8];
    for (int j = 0; j < 8; j++) m[j] = lg[0];
    int64_t v = 0;
    for (; v + 8 <= n; v += 8)
        for (int j = 0; j < 8; j++) m[j] = m[j] < lg[v + j] ? lg[v + j] : m[j];   // NaN on the right: m kept
    for (; v < n; v++) m[0] = m[0] < lg[v] ? lg[v] : m[0];
    float mx = m[0];
    for (int j = 1; j < 8; j++) mx = mx < m[j] ? m[j] : mx;
    for (int64_t i = 0; i < n; i++) if (lg[i] == mx) return (int) i;
    return 0;   // only when lg[0] is NaN (then mx is NaN): the reference loop returns 0 too
}
