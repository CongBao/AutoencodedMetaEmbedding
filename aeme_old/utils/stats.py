# statistical functions

from __future__ import division, print_function

import math

from scipy import stats

def r_confidence_interval(r, n, alpha=0.05): # correlation
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = math.tanh(z - z_crit * se)
    hi = math.tanh(z + z_crit * se)
    return lo, hi

def clopper_pearson(r, n, alpha=0.05): # accuracy
    k = round(n * r)
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi
