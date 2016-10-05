import testbed1 as testbed
from math import sqrt
import numpy as np
from functools import partial


# Evan Miller's Sequential stopping rule

def sequential_evanmiller_onesided(a_arm, b_arm):
    n = 10000
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
        return None, None


def expected_loss_plus(a_arm, b_arm):
    if (sum(a_arm.counts)+sum(b_arm.counts)) % 1000 == 0:
        mrr = [5, 9, 30, 0]
        # Run 100000 test and simulate the loss
        priors = np.array([1, 1, 1, 1])
        a_results = (np.random.dirichlet(a_arm.counts + priors, 10000) * mrr).sum(axis=1)
        b_results = (np.random.dirichlet(b_arm.counts + priors, 10000) * mrr).sum(axis=1)

        if sum(a_results < b_results) / len(a_results) > .999:
            return 2, None
        elif sum(b_results < a_results) / len(a_results) > .999:
            return 1, None

        expected_loss = np.maximum(a_results - b_results, 0).mean()
        expected_benefit = np.maximum(b_results - a_results, 0).mean()

        if (expected_loss < expected_benefit) and (expected_loss < .001):
            return 2, None
        if sum(a_arm.counts) + sum(b_arm.counts) >= 50000:
            if sum(a_results < b_results)/len(a_results) > .50:
                return 2, None
            else:
                return 1, None

        if (expected_loss > expected_benefit) and expected_benefit < .001:
            return 1, None
        return None, None

    return None, None


def expected_loss_pure(a_arm, b_arm):
    if (sum(a_arm.counts)+sum(b_arm.counts)) >= 10000:
        if (sum(a_arm.counts)+sum(b_arm.counts)) % 100 == 0:
            mrr = [5, 9, 30, 0]
            # Run 100000 test and simulate the loss
            priors = np.array([1, 1, 1, 1])
            a_results = (np.random.dirichlet(a_arm.counts + priors, 10000) * mrr).sum(axis=1)
            b_results = (np.random.dirichlet(b_arm.counts + priors, 10000) * mrr).sum(axis=1)

            expected_loss = np.maximum(a_results - b_results, 0).mean()
            expected_benefit = np.maximum(b_results - a_results, 0).mean()

            if (expected_loss < expected_benefit) and (expected_loss < .001):
                return 2, None
            if sum(a_arm.counts) + sum(b_arm.counts) >= 100000:
                return 1, None

            return None, None

    return None, None


def certainty_99_or_100000(a_arm, b_arm):
    if (sum(a_arm.counts)+sum(b_arm.counts)) % 1000 == 0:
        mrr = [5, 9, 30, 0]
        priors = np.array([1, 1, 1, 1])
        a_results = (np.random.dirichlet(a_arm.counts + priors, 10000) * mrr).sum(axis=1)
        b_results = (np.random.dirichlet(b_arm.counts + priors, 10000) * mrr).sum(axis=1)

        if sum(a_results < b_results) / len(a_results) > .999:
            return 2, None
        elif sum(b_results < a_results) / len(a_results) > .995:
            return 1, None
        elif sum(a_arm.counts) + sum(b_arm.counts) >= 200000:
            if sum(a_results < b_results)/len(a_results) > .75:
                return 2, None
            else:
                return 1, None
        else:
            return None, None
    else:
        return None, None


# Pick a winner the first time significance is reached

def first_significant(a_arm, b_arm):
    alpha = 0.1
    max_samples = 10000
    n_a = a_arm.total_samples()
    n_b = b_arm.total_samples()
    s_a = a_arm.total_conversions()
    s_b = b_arm.total_conversions()
    
    if (s_a + s_b > 100):
        if testbed.get_p_value(s_a, n_a, s_b, n_b) < alpha:
            if (s_a / n_a) > (s_b / n_b):
                return 1, None
            else:
                return 2, None
        elif n_a + n_b >= max_samples:
            return 1, None
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


def thompson_sampling(a_arm, b_arm):
    mrr = [5, 9, 30, 0]
    threshold = 0.95
    max_samples = 50000

    a_prior = np.array([1, 1, 1, 1])
    b_prior = a_prior

    # Run 100000 test simulations to get the probability that B is better than A, only every x samples
    if (a_arm.total_samples() + b_arm.total_samples()) % 100 == 0:

        p_b_optimal = testbed.get_p_b_optimal(a_arm, b_arm)

        # print("Probability that B is optimal:" + str(p_b_optimal))

        if p_b_optimal > threshold:
            return 2, None
        elif p_b_optimal < (1 - threshold):
            return 1, None
        elif a_arm.total_samples() + b_arm.total_samples() > max_samples:
            if p_b_optimal > 0.95:
                return 2, None
            else:
                return 1, None
        else:
            return None, (1-p_b_optimal)
    else:
        return None, None


def thompson_sampling_isaac(a_arm, b_arm):
    mrr = [5, 9, 30, 0]
    threshold = 0.95
    max_samples = 100000

    a_prior = np.array([1, 1, 1, 1])
    b_prior = a_prior

    # Run 100000 test simulations to get the probability that B is better than A, only every x samples
    if (a_arm.total_samples() + b_arm.total_samples()) % 100 == 0:
        a_results = (np.random.dirichlet(np.array(a_arm.counts) + a_prior, 10000) * mrr).sum(axis=1)
        b_results = (np.random.dirichlet(np.array(b_arm.counts) + b_prior, 10000) * mrr).sum(axis=1)

        p_b_optimal = sum(b_results > a_results)/len(a_results)

        # print("Probability that B is optimal:" + str(p_b_optimal))

        if p_b_optimal > threshold:
            return 2, None
        elif p_b_optimal < (1 - threshold):
            return 1, None
        elif a_arm.total_samples() + b_arm.total_samples() > max_samples:
            if p_b_optimal > 0.55:
                return 2, None
            else:
                return 1, None
        else:
            return None, (1-p_b_optimal)
    else:
        return None, None


def super_improvement(a_arm, b_arm):
    total = a_arm.total_samples() + b_arm.total_samples()

    if total == 200000:
        prior = [1, 1, 1, 1]
        mrr = [5, 9, 30, 0]
        a_results = (np.random.dirichlet(np.array(a_arm.counts) + prior, 10000) * mrr).sum(axis=1)
        b_results = (np.random.dirichlet(np.array(b_arm.counts) + prior, 10000) * mrr).sum(axis=1)
        p_b_optimal = sum(b_results > a_results) / len(a_results)
        if p_b_optimal > .75:
            return 2, None
        return 1, None
    return None, None

class SpencerHelper:
    def __init__(self):
        self.check_day_three = False
        self.check_day_seven = False
        self.check_week_three = False

    def set_checker_three(self):
        self.check_day_three = True

    def clear_checker_three(self):
        self.check_day_three = False

    def set_checker_seven(self):
        self.check_day_seven = True


    def clear_checker_seven(self):
        self.check_day_seven = False

    def set_week_three(self):
        self.check_week_three = True

    def clear_week_three(self):
        self.check_week_three = False


def spencer_rule(a_arm, b_arm, helper):
    def get_p_spencer():
        return testbed.get_p_value(a_arm.total_conversions(),
                                   a_arm.total_samples(),
                                   b_arm.total_conversions(),
                                   b_arm.total_samples())

    def get_cash(multiplier = 1):
        mrr = [5, 9, 30, 0]
        priors = np.array([1, 1, 1, 1])
        a_results = (np.random.dirichlet(a_arm.counts + priors, 10000) * mrr).sum(axis=1)
        b_results = (np.random.dirichlet(b_arm.counts + priors, 10000) * mrr).sum(axis=1)
        if b_results.mean() > a_results.mean():
            return 'positive'
        else:
            return 'negative'

    def difference_of_conversion_rate():
        if a_arm.total_conversions() > 0:
            return b_arm.total_conversions() / a_arm.total_conversions() - 1
        else:
            return 0

    def team_check():
        if a_arm.counts[2] + b_arm.counts[2] >= 80:
            if b_arm.counts[2] / a_arm.counts[2] - 1 < -.2 and testbed.get_p_value(a_arm.counts[2], sum(a_arm.counts), b_arm.counts[2], sum(b_arm.counts)) < .1:
                return 1
            if testbed.get_p_value(a_arm.counts[2], sum(a_arm.counts), b_arm.counts[2], sum(b_arm.counts)) < .1 and get_cash(1.1) == 'mpositive' and b_arm.counts[2] / a_arm.counts[2] - 1 > 0:
                return 2
        return 0



    day_count = 400
    participant_count = a_arm.total_samples() + b_arm.total_samples()


    # Check for a drop in conversion rates (95% certainty - use 90% two-sided). If so, revert to T-A. If there's a 5%
    # drop that isn't significant, tell it to check on day three
    if participant_count / day_count == 1:

        # Check for Team Overrides
        team_override = team_check()
        if team_override == 1:
            return 1, None
        elif team_override == 2:
            return 2, None

        if get_p_spencer() < .1 and difference_of_conversion_rate() < 0 and get_cash() == 'negative':
            return 1, None
        elif difference_of_conversion_rate() < -.05:
            helper.set_checker_three()

    # On day 3 check for a continued 5% drop - if it's there, check and call (if needed) on day 7
    if participant_count / day_count == 3 and helper.check_day_three:
        helper.clear_checker_three()

        # Check for Team Overrides
        team_override = team_check()
        if team_override == 1:
            return 1, None
        elif team_override == 2:
            return 2, None

        if difference_of_conversion_rate() < -.05:  # TODO: is this supposed to be 0?
            helper.set_checker_seven()

    # Day 7 - Call it for T-A if conversion rate is down 5%+ and if certainty is above 80% and value is < 0
    if participant_count / day_count == 7 and helper.check_day_seven:
        helper.clear_checker_seven()

        # Check for Team Overrides
        team_override = team_check()
        if team_override == 1:
            return 1, None
        elif team_override == 2:
            return 2, None

        if difference_of_conversion_rate() < -.05 and get_p_spencer() < .4 and get_cash() == 'negative':  # TODO: is this supposed to be 0?
            return 1, None

    # Check week 2: If payment rate is up w/ 95% certainty and value is positive, T-B. Else, keep going
    if participant_count / day_count == 14:

        # Check for Team Overrides
        team_override = team_check()
        if team_override == 1:
            return 1, None
        elif team_override == 2:
            return 2, None

        if difference_of_conversion_rate() > 0 and get_p_spencer() < .1 and get_cash() == 'positive':
            return 2, None
        elif difference_of_conversion_rate() > 0 and get_p_spencer() < .2 and get_cash() == 'positive':
            helper.set_week_three()

    # Check week 3 if needed
    if participant_count / day_count == 21:
        helper.clear_week_three()

        # Check for Team Overrides
        team_override = team_check()
        if team_override == 1:
            return 1, None
        elif team_override == 2:
            return 2, None

        if difference_of_conversion_rate() > 0 and get_p_spencer() < .1 and get_cash() == 'positive':
            return 2, None

    # Check week 4
    if participant_count / day_count == 28:

        # Check for Team Overrides
        team_override = team_check()
        if team_override == 1:
            return 1, None
        elif team_override == 2:
            return 2, None

        if difference_of_conversion_rate() > 0 and get_p_spencer() < .1 and get_cash() == 'positive':
            return 2, None
        elif (difference_of_conversion_rate() > 0 and get_p_spencer() < .4 and get_cash() == 'positive') == False:
            return 1, None

    # Check week 6 and 8
    if participant_count / day_count == 42 or participant_count / day_count == 56:

        # Check for Team Overrides
        team_override = team_check()
        if team_override == 1:
            return 1, None
        elif team_override == 2:
            return 2, None

        if difference_of_conversion_rate() > 0 and get_p_spencer() < .1 and get_cash() == 'positive':
            return 2, None
        elif (difference_of_conversion_rate() > .03 and get_p_spencer() < .4 and get_cash() == 'positive') == False:
            return 1, None

    # Check week 10
    if participant_count / day_count == 70:

        # Check for Team Overrides
        team_override = team_check()
        if team_override == 1:
            return 1, None
        elif team_override == 2:
            return 2, None

        if get_p_spencer() < .2 and get_cash(1.05) == 'positive':
            return 2, None
        else:
            return 1, None

    return None, None


if __name__ == '__main__':
    my_helper = SpencerHelper()
    test_partial = partial(spencer_rule, helper=my_helper)
    testbed.multi_test([test_partial], max_tests=1000, plot=True, seed=5)

