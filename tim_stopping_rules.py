import testbed1 as testbed
from math import sqrt, log
import numpy as np

'''Every test rule needs to accept two arm objects from the test-runner. The methods available to arm abjects are:
total_conversions: The count of conversions for the arm
total_samples: The total number of conversions for the arm
conversion_rate: total_conversions/total_samples

Every test rule should return a two-element tuple. The first element should be 1 to call A the winner, 2 to call B the
winner, or None to continue the test. THe second element should be the proportion of people that you want to go into the
A arm, or None for a 50/50 split.

Each arm also has a 'counts' attribute which is a list of the counts of conversions at each level. It can be thought of
as [basic, pro, team, free] levels. For the purpose of the testing, we are using MRR values of [5, 9, 30, 0] for the
tests.

the testbed1 module also includes the helper function
get_p_value(a_success, a_total, b_success, b_total, alternative='two sided') which will give the p-value for any set of
numbers. Be sure to specify if you would like a "one sided" test rather than a 'two sided' test.
'''


def sequential_evanmiller_onesided(a_arm, b_arm):
    n = 10000
    t = b_arm.total_conversions()
    c = a_arm.total_conversions()

    if t - c >= 2 * sqrt(n):
        return 2, None
    elif t + c >= n:
        return 1, None
    else:
        return None, None


def sequential_evanmiller_twosided(a_arm, b_arm):
    n = 50
    t = b_arm.total_conversions()
    c = a_arm.total_conversions()
    mrr = [5, 9, 30, 0]

    if (t - c) >= 2.25 * sqrt(n):
        return 2, None
    elif (c - t) >= 2.25 * sqrt(n):
        return 1, None
    elif (t + c) >= n:
        if (np.array(a_arm.counts) * mrr).sum() > (np.array(b_arm.counts) * mrr).sum():
            return 1, None
        else:
            return 2, None
    else:
        return None, None

def em_200(a_arm, b_arm):
    n = 200
    t = b_arm.total_conversions()
    c = a_arm.total_conversions()
    mrr = [5, 9, 30, 0]

    if (t - c) >= 2.25 * sqrt(n):
        return 2, None
    elif (c - t) >= 2.25 * sqrt(n):
        return 1, None
    elif (t + c) >= n:
        if (np.array(a_arm.counts) * mrr).sum() > (np.array(b_arm.counts) * mrr).sum():
            return 1, None
        else:
            return 2, None
    else:
        return None, None

def em_50(a_arm, b_arm):
    n = 50
    t = b_arm.total_conversions()
    c = a_arm.total_conversions()
    mrr = [5, 9, 30, 0]

    if (t - c) >= 2.25 * sqrt(n):
        return 2, None
    elif (c - t) >= 2.25 * sqrt(n):
        return 1, None
    elif (t + c) >= n:
        if (np.array(a_arm.counts) * mrr).sum() > (np.array(b_arm.counts) * mrr).sum():
            return 1, None
        else:
            return 2, None
    else:
        return None, None

def em_45(a_arm, b_arm):
    n = 45
    t = b_arm.total_conversions()
    c = a_arm.total_conversions()
    mrr = [5, 9, 30, 0]

    if (t - c) >= 2.25 * sqrt(n):
        return 2, None
    elif (c - t) >= 2.25 * sqrt(n):
        return 1, None
    elif (t + c) >= n:
        if (np.array(a_arm.counts) * mrr).sum() > (np.array(b_arm.counts) * mrr).sum():
            return 1, None
        else:
            return 2, None
    else:
        return None, None

def em_40(a_arm, b_arm):
    n = 40
    t = b_arm.total_conversions()
    c = a_arm.total_conversions()
    mrr = [5, 9, 30, 0]

    if (t - c) >= 2.25 * sqrt(n):
        return 2, None
    elif (c - t) >= 2.25 * sqrt(n):
        return 1, None
    elif (t + c) >= n:
        if (np.array(a_arm.counts) * mrr).sum() > (np.array(b_arm.counts) * mrr).sum():
            return 1, None
        else:
            return 2, None
    else:
        return None, None

# Pick a winner the first time significance is reached

def first_significant_two_sided(a_arm, b_arm):
    alpha = 0.2
    max_samples = 5000
    n_a = a_arm.total_samples()
    n_b = b_arm.total_samples()
    s_a = a_arm.total_conversions()
    s_b = b_arm.total_conversions()
    mrr = [5, 9, 30, 0]

    if (s_a + s_b > 10):
        if testbed.get_p_value(s_a, n_a, s_b, n_b) < alpha:
            if (np.array(a_arm.counts) * mrr).sum() > (np.array(b_arm.counts) * mrr).sum():
                return 1, None
            else:
                return 2, None
        elif n_a + n_b >= max_samples:
            if (np.array(a_arm.counts) * mrr).sum() > (np.array(b_arm.counts) * mrr).sum():
                return 1, None
            else:
                return 2, None
        else:
            return None, None
    else:
        return None, None


# Only test if B is better than A, one-sided test

def first_significant_one_sided(a_arm, b_arm):
    alpha = 0.1
    max_samples = 100000
    n_a = a_arm.total_samples()
    n_b = b_arm.total_samples()
    s_a = a_arm.total_conversions()
    s_b = b_arm.total_conversions()

    if (s_a + s_b > 50):
        if testbed.get_p_value(s_a, n_a, s_b, n_b, alternative="greater") < alpha:
            return 2, None
        elif n_a + n_b >= max_samples:
            return 1, None
        else:
            return None, None
    else:
        return None, None


# if it looks like B might be worse, call the test early, otherwise wait for significance

def first_significant_conservative(a_arm, b_arm):
    alpha_1 = 0.25
    alpha_2 = 0.1
    max_samples = 100000
    n_a = a_arm.total_samples()
    n_b = b_arm.total_samples()
    s_a = a_arm.total_conversions()
    s_b = b_arm.total_conversions()

    if (s_a + s_b > 20):
        if testbed.get_p_value(s_a, n_a, s_b, n_b, alternative="less") < alpha_1:
            return 1, None
        elif testbed.get_p_value(s_a, n_a, s_b, n_b, alternative="greater") < alpha_2:
            return 2, None
        elif n_a + n_b >= max_samples:
            return 1, None
        else:
            return None, None
    else:
        return None, None


# stop after a fixed number of samples

def fixed_sample(a_arm, b_arm):
    n_a = a_arm.total_samples()
    n_b = b_arm.total_samples()
    s_a = a_arm.total_conversions()
    s_b = b_arm.total_conversions()
    sample_size = 1000
    mrr = [5, 9, 30, 0]

    if (n_a + n_b >= sample_size):
        if (np.array(a_arm.counts) * mrr).sum() > (np.array(b_arm.counts) * mrr).sum():
            return 1, None
        else:
            return 2, None
    else:
        return None, None

def fixed_sample2(a_arm, b_arm):
    alpha = 0.1
    n_a = a_arm.total_samples()
    n_b = b_arm.total_samples()
    s_a = a_arm.total_conversions()
    s_b = b_arm.total_conversions()
    sample_size = 80000
    mrr = [5, 9, 30, 0]

    if (n_a + n_b >= sample_size):
        if (np.array(a_arm.counts) * mrr).sum() > (np.array(b_arm.counts) * mrr).sum():
            return 1, None
        else:
            return 2, None
    else:
        return None, None


def thompson2(a_arm, b_arm):
    mrr = [5, 9, 30, 0]
    threshold = 0.8
    max_samples = 50000

    a_prior = np.array([1, 1, 1, 1])
    b_prior = a_prior

    if (a_arm.total_samples() + b_arm.total_samples()) % 100 == 0:

        p_b_optimal = testbed.get_p_b_optimal(a_arm, b_arm)

        # print("Probability that B is optimal:" + str(p_b_optimal))

        if p_b_optimal > threshold:
            return 2, None
        elif p_b_optimal < (1 - threshold):
            return 1, None
        elif a_arm.total_samples() + b_arm.total_samples() >= max_samples:
            if p_b_optimal > 0.5:
                return 2, None
            else:
                return 1, None
        else:
            return None, (1 - p_b_optimal)
    else:
        return None, None

def thompson_sampling2(a_arm, b_arm):
    mrr = [5, 9, 30, 0]
    threshold = 0.75
    max_samples = 50000

    a_prior = np.array([1, 1, 1, 1])
    b_prior = a_prior

    total_samples = a_arm.total_samples() + b_arm.total_samples()

    if total_samples > 100 and total_samples % 10 == 0:

        p_b_optimal = testbed.get_p_b_optimal(a_arm, b_arm)

        # print("Probability that B is optimal:" + str(p_b_optimal))

        if p_b_optimal > threshold:
            return 2, None
        elif p_b_optimal < (1 - threshold):
            return 1, None
        elif a_arm.total_samples() + b_arm.total_samples() >= max_samples:
            if p_b_optimal > 0.5:
                return 2, None
            else:
                return 1, None
        else:
            return None, (1 - p_b_optimal)
    else:
        return None, None

def tim_thompson_sampling(a_arm, b_arm):
    mrr = [5, 9, 30, 0]
    threshold = 0.75
    max_samples = 50000

    a_prior = np.array([1, 1, 1, 1])
    b_prior = a_prior

    total_samples = a_arm.total_samples() + b_arm.total_samples()

    if total_samples % 100 == 0:

        p_b_optimal = testbed.get_p_b_optimal(a_arm, b_arm)

        # print("Probability that B is optimal:" + str(p_b_optimal))

        if p_b_optimal > threshold:
            return 2, None
        elif p_b_optimal < (1 - threshold):
            return 1, None
        elif a_arm.total_samples() + b_arm.total_samples() >= max_samples:
            if p_b_optimal > 0.5:
                return 2, None
            else:
                return 1, None
        else:
            return None, (1 - p_b_optimal)
    else:
        return None, None


def optimizely(a_arm, b_arm):

    if a_arm.total_conversions() > 1 and b_arm.total_conversions() > 1:
        Xbar = a_arm.conversion_rate()
        Ybar = b_arm.conversion_rate()
        theta = Ybar - Xbar
        tau = .001
        alpha = 0.1

        V_n = 2 * (Xbar * (1 - Xbar) + Ybar * (1 - Ybar)) / (a_arm.total_samples() + b_arm.total_samples())

        threshold = sqrt((2 * log(1 / alpha) - log(V_n / V_n + tau)) * (V_n*(V_n + tau)/tau))

        print(str(threshold))

        if abs(theta) > threshold:
            if Ybar > Xbar:
                return 2, None
            else:
                return 1, None
        else:
            return None, None
    else:
        return None, None


def first_to_10(a_arm, b_arm):

    x = 10

    if a_arm.total_conversions() > x:
        return 1, None
    elif b_arm.total_conversions() > x:
        return 2, None

    return None, None

def first_to_20(a_arm, b_arm):

    x = 20

    if a_arm.total_conversions() > x:
        return 1, None
    elif b_arm.total_conversions() > x:
        return 2, None

    return None, None

def first_to_50(a_arm, b_arm):

    x = 50

    if a_arm.total_conversions() > x:
        return 1, None
    elif b_arm.total_conversions() > x:
        return 2, None

    return None, None

def first_to_100(a_arm, b_arm):

    x = 100

    if a_arm.total_conversions() > x:
        return 1, None
    elif b_arm.total_conversions() > x:
        return 2, None

    return None, None

def first_to_15(a_arm, b_arm):

    x = 15

    if a_arm.total_conversions() > x:
        return 1, None
    elif b_arm.total_conversions() > x:
        return 2, None

    return None, None

def first_to_25(a_arm, b_arm):

    x = 25

    if a_arm.total_conversions() > x:
        return 1, None
    elif b_arm.total_conversions() > x:
        return 2, None

    return None, None


def bayes_nopriors(a_arm, b_arm):
    threshold = 0.75
    max_samples = 50000

    if (a_arm.total_conversions() + b_arm.total_conversions()) % 100:

        p_b_optimal = testbed.get_p_b_optimal(a_arm, b_arm)

        if p_b_optimal > threshold:
            return 2, None
        elif p_b_optimal < (1 - threshold):
            return 1, None
        elif a_arm.total_samples() + b_arm.total_samples() >= max_samples:
            if p_b_optimal > 0.5:
                return 2, None
            else:
                return 1, None
        else:
            return None, None
    else:
        return None, None

def bayes_priors(a_arm, b_arm):

    threshold = 0.8
    max_samples = 50000

    if (a_arm.total_samples() + b_arm.total_samples()) % 100 == 0:

        p_b_optimal = testbed.get_p_b_optimal(a_arm, b_arm, a_priors=[10, 8.2, 2.5, 979], b_priors=[9.5, 7.8, 2.4, 980])

        if p_b_optimal > threshold:
            return 2, None
        elif p_b_optimal < (1 - threshold):
            return 1, None
        elif a_arm.total_samples() + b_arm.total_samples() >= max_samples:
            if p_b_optimal > 0.5:
                return 2, None
            else:
                return 1, None
        else:
            return None, None
    else:
        return None, None

def first_conversion(a_arm, b_arm):
    if b_arm.total_conversions() > 0:
        return 2, None

    return 1, None

def best_after_100(a_arm, b_arm):
    n =100
    mrr = [5, 9, 30, 0]

    if a_arm.total_samples() + b_arm.total_samples() >= n:
        if (np.array(a_arm.counts) * mrr).sum() > (np.array(b_arm.counts) * mrr).sum():
            return 1, None
        else:
            return 2, None
    else:
        return None, None

def dynamic_threshold(a_arm, b_arm):
    #alpha = 0.1        .34 - .44
    #max_samples = 10000

    #alpha = 0.1
    #max_samples = 8000     .34 - .56

    #alpha = 0.1            .36 - .63
    #max_samples = 6000

    #alpha = 0.1                 .23 - .72
    #max_samples = 5000

    #alpha = 0.1                 .33 - .77
    #max_samples = 4000

    #alpha = 0.2            .4 - .89
    #max_samples = 4000

    alpha = 0.3
    max_samples = 3000

    n_a = a_arm.total_samples()
    n_b = b_arm.total_samples()
    s_a = a_arm.total_conversions()
    s_b = b_arm.total_conversions()
    mrr = [5, 9, 30, 0]

    # wait until we have at least one conversion on each arm
    if s_a > 1 and s_b > 1:

        # use a linear p-value threshold
        if testbed.get_p_value(s_a, n_a, s_b, n_b) < (alpha/max_samples) * (n_a + n_b):

            # if the test is statistically significant, pick the arm with the highest value
            if (np.array(a_arm.counts) * mrr).sum() > (np.array(b_arm.counts) * mrr).sum():
                return 1, None
            else:
                return 2, None

        # if we reach the max number of samples, stop the test
        elif n_a + n_b >= max_samples:
            if (np.array(a_arm.counts) * mrr).sum() > (np.array(b_arm.counts) * mrr).sum():
                return 1, None
            else:
                return 2, None
        else:
            return None, None
    else:
        return None, None


if __name__ == '__main__':
    ''' This is where you actually run the stopping rules. The first arg is a list of the rules that you want to test.
    max_tests is the number of tests that you want to run for each arm. plot plots the output. Seed is the seed for the
    random test generator (so that you can compare each stopping rule using the same test data). A seed of False or None
    won't set any seed, but any other value will be used for the seed itself.

    The resulting data/graphs will be stored under the results folder on your local machine. '''

    testbed.multi_test([dynamic_threshold,dynamic_threshold,dynamic_threshold,dynamic_threshold,dynamic_threshold,dynamic_threshold], max_tests=500, plot=False, seed=None)
