import testbed1 as testbed
from math import sqrt
import numpy as np


# Evan Miller's Sequential stopping rule

def sequential_evanmiller_onesided(a_arm, b_arm):
    n = 1000
    t = b_arm.total_conversions()
    c = a_arm.total_conversions()

    if t-c >= 2*sqrt(n):
        return 2, None
    elif t+c >= n:
        return 1, None
    else:
        return None, None


def sequential_evanmiller_twosided(a_arm, b_arm):
    n = 1000
    t = b_arm.total_conversions()
    c = a_arm.total_conversions()

    if (t-c) >= 2.25*sqrt(n):
        return 2, None
    elif (c-t) >= 2.25*sqrt(n):
        return 1, None
    elif (t+c) >= n:
        return 1, None
    else:
        return None


def expected_loss_test(a_arm, b_arm):
    mrr = [5, 9, 30, 0]
    # Run 100000 test and simulate the loss
    a_results = np.random.dirichlet(a_arm, 100000) * mrr
    b_results = np.random.dirichlet(b_arm, 100000) * mrr

    expected_loss = np.maximum(a_results - b_results, 0).mean()
    expected_benefit = np.maximum(b_results - a_results, 0).mean()
    

# Pick a winner the first time significance is reached

def first_significant(a_arm, b_arm):
    alpha = 0.1
    max_samples = 10000
    n_a = a_arm.total_samples()
    n_b = b_arm.total_samples()
    s_a = a_arm.total_conversions()
    s_b = b_arm.total_conversions()
    
    if (s_a + s_b > 100):
        if (testbed.get_p_value(s_a, n_a, s_b, n_b) < alpha or n_a + n_b >= max_samples):
            if (s_a / n_a > s_b / n_b):
                return 1, None
            else:
                return 2, None
        else:
            return None, None
    else:
        return None, None

# stop after a fixed number of samples

def fixed_sample(a_arm, b_arm):
    alpha = 0.1
    n_a = a_arm.total_samples()
    n_b = b_arm.total_samples()
    s_a = a_arm.total_conversions()
    s_b = b_arm.total_conversions()
    sample_size = 100000
    
    if (n_a + n_b >= sample_size):
        if (testbed.get_p_value(s_a, n_a, s_b, n_b) < alpha):
            if (s_a / n_a > s_b / n_b):
                return 1, None
            else:
                return 2, None
        else:
            return 1, None
    else:
        return None, None


agg_results, _ = testbed.multi_test([sequential_evanmiller_onesided], max_tests=1000, plot=True)
agg_results.to_csv('agg_results.csv')

# print(agg_results)
# print(_)

