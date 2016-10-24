import numpy as np
from functools import partial
import testbed1 as testbed
from math import floor, sqrt


# Max sample or 99% certain (with test size normalization based on max people)
class IsaacHelper:
    def __init__(self):
        self.max_people = 460000000
        self.max_tests = 1300
        self.remaining_tests = self.max_tests
        self.remaining_people = self.max_people

        self.this_test = floor(self.remaining_people/self.remaining_tests)

    def finish_test(self):
        self.remaining_tests -= 1
        self.this_test = floor(self.remaining_people/self.remaining_tests)

    def add_sample(self):
        self.max_people -= 1


def corn_flakes(a_arm, b_arm, m_helper):
    m_helper.add_sample()
    people_count = a_arm.total_samples() + b_arm.total_samples()
    if people_count >= m_helper.this_test:
        m_helper.finish_test()
        return (2, None) if testbed.get_p_b_optimal(a_arm, b_arm) > .6 else (1, None)

    if people_count % floor(m_helper.this_test / 4) == 0:
        if testbed.get_p_b_optimal(a_arm, b_arm) > .99:
            m_helper.finish_test()
            return 2, None
        elif testbed.get_p_b_optimal(a_arm, b_arm) < .01:
            m_helper.finish_test()
            return 1, None
        else:
            return None, None
    else:
        return None, None


def parfait(a_arm, b_arm):
    mrr = [5, 9, 30, 0]
    threshold = 0.6
    max_samples = 353846

    # Run 100000 test simulations to get the probability that B is better than A, only every x samples
    if (a_arm.total_samples() + b_arm.total_samples()) % 10000 == 0:

        p_b_optimal = testbed.get_p_b_optimal(a_arm, b_arm)

        # print("Probability that B is optimal:" + str(p_b_optimal))

        if a_arm.total_samples() + b_arm.total_samples() > max_samples - 9999:
            if p_b_optimal > threshold:
                return 2, None
            else:
                return 1, None
        else:
            return None, (1-p_b_optimal)
    else:
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

    def get_cash(multiplier=1):
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
            if b_arm.counts[2] / a_arm.counts[2] - 1 < -.2 and testbed.get_p_value(a_arm.counts[2], sum(a_arm.counts),
                                                                                   b_arm.counts[2],
                                                                                   sum(b_arm.counts)) < .1:
                return 1
            if testbed.get_p_value(a_arm.counts[2], sum(a_arm.counts), b_arm.counts[2],
                                   sum(b_arm.counts)) < .1 and get_cash(1.1) == 'mpositive' and b_arm.counts[2] / \
                    a_arm.counts[2] - 1 > 0:
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


def KanyeWest(a_arm, b_arm):
    from scipy.stats import norm
    n = a_arm.total_samples() + b_arm.total_samples()
    k = 5000000
    nTwo = 0
    taco = 1
    stop = norm.ppf(n/(k+(n*2))*sqrt(n))*-1
    if a_arm.total_conversions() >=taco and b_arm.total_conversions() >= taco and abs(a_arm.conversion_rate()/b_arm.conversion_rate()) > stop and a_arm.conversion_rate() > b_arm.conversion_rate():
        return 1, None
    if a_arm.total_conversions() >=taco and b_arm.total_conversions() >= taco and abs(a_arm.conversion_rate()/b_arm.conversion_rate()) > stop and a_arm.conversion_rate() < b_arm.conversion_rate():
        return 2, None
    k = k - (a_arm.total_samples() + b_arm.total_samples())
    if k < 5000:
        return 1, None
    n = a_arm.total_samples() + b_arm.total_samples()
    k = k - n
    return None, None


def KRSone(a_arm, b_arm):
    # After 100 conversions on B, check if it's 5% better, but if A ever becomes 10% worse, stop it.
    if b_arm.total_conversions() >= 100 and b_arm.conversion_rate() / a_arm.conversion_rate() - 1 > 0.05:
        return 2, None
    if a_arm.total_conversions() > 1 and b_arm.total_conversions() > 1 and b_arm.conversion_rate() / a_arm.conversion_rate() - 1 < 0.05:
        return 1, None
    return None, None


if __name__ == '__main__':
    ''' This is where you actually run the stopping rules. The first arg is a list of the rules that you want to test.
    max_tests is the number of tests that you want to run for each rule. plot plots the output. Seed is the seed for the
    random test generator (so that you can compare each stopping rule using the same test data). A seed of False or None
    won't set any seed, but any other value will be used for the seed itself. Max_people represents the maximum number
    of people that can be tested. Test_size is the time over which the value of the test will be measured.

    The resulting data/graphs will be stored under the results folder on your local machine. '''

    if __name__ == '__main__':
        my_helper = SpencerHelper()
        test_partial = partial(spencer_rule, helper=my_helper)
        isaac_helper = IsaacHelper()
        isaac_partial = partial(corn_flakes, m_helper=isaac_helper)
        testbed.multi_test([test_partial, isaac_partial, parfait, KanyeWest, KRSone], max_tests=1300, plot=False,
                           max_people=460000000, test_size=5000000, seed=564654654)
