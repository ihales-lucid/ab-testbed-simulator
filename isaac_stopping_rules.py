import testbed1 as testbed
from math import sqrt
import numpy as np
from functools import partial, update_wrapper


def certainty_or_count(a_arm, b_arm, certainty=.999, count=25000, final_certainty=.9):
    if (sum(a_arm.counts) + sum(b_arm.counts)) % 1000 == 0:
        mrr = [5, 9, 30, 0]
        priors = np.array([1, 1, 1, 1])
        a_results = (np.random.dirichlet(a_arm.counts + priors, 10000) * mrr).sum(axis=1)
        b_results = (np.random.dirichlet(b_arm.counts + priors, 10000) * mrr).sum(axis=1)

        if sum(a_results < b_results) / len(a_results) > certainty:
            return 2, None
        elif sum(b_results < a_results) / len(a_results) > certainty:
            return 1, None
        elif sum(a_arm.counts) + sum(b_arm.counts) >= count:
            if sum(a_results < b_results) / len(a_results) > final_certainty:
                return 2, None
            else:
                return 1, None
        else:
            return None, None
    else:
        return None, None


def func7(a_arm, b_arm):
    return certainty_or_count(a_arm, b_arm, count=25000, final_certainty=.9)


class CurrentRuleHelper:
    def __init__(self):
        self.good_count = 0
        self.bad_count = 0

    def is_good(self):
        self.good_count += 1

    def is_bad(self):
        self.bad_count += 1

    def not_good(self):
        self.good_count = 0

    def not_bad(self):
        self.bad_count = 0


def current_rule(a_arm, b_arm, helper):
    daily_views = 400

    # For the first few days, check if there's 95% certainty that it's worse
    if (a_arm.total_samples() + b_arm.total_samples()) / daily_views in [1, 2, 3]:
        if a_arm.conversion_rate() > b_arm.conversion_rate() and testbed.get_p_value(a_arm.total_conversions(),
                                                                                     a_arm.total_samples(),
                                                                                     b_arm.total_conversions(),
                                                                                     b_arm.total_samples()) < 0.1:
            helper.not_bad()
            helper.not_good()
            return 1, None
    # Check every two weeks
    if (a_arm.total_samples() + b_arm.total_samples()) % (daily_views * 14) == 0:

        # Check for 2 weeks of significance in payments in either Direction:
        if a_arm.conversion_rate() < b_arm.conversion_rate() and testbed.get_p_value(a_arm.total_conversions(),
                                                                                     a_arm.total_samples(),
                                                                                     b_arm.total_conversions(),
                                                                                     b_arm.total_samples()) < 0.14:
            helper.is_good()
            if helper.good_count >= 2:
                helper.not_bad()
                helper.not_good()
                return 2, None

        if a_arm.conversion_rate() > b_arm.conversion_rate() and testbed.get_p_value(a_arm.total_conversions(),
                                                                                     a_arm.total_samples(),
                                                                                     b_arm.total_conversions(),
                                                                                     b_arm.total_samples()) < 0.10:
            helper.is_bad()
            if helper.bad_count >= 2:
                helper.not_bad()
                helper.not_good()
                return 2, None
        else:
            helper.not_good()
            helper.not_bad()

        # If B is 93% certain in payment amounts and B values are up by 2.5%, call it a win. If A's down 80% and payment
        # amounts are down at all, call it a loss.

        if a_arm.conversion_rate() < b_arm.conversion_rate() and testbed.get_p_value(a_arm.total_conversions(),
                                                                                     a_arm.total_samples(),
                                                                                     b_arm.total_conversions(),
                                                                                     b_arm.total_samples()) < 0.14:

            mrr = [5, 9, 30, 0]
            priors = np.array([1, 1, 1, 1])
            a_results = (np.random.dirichlet(a_arm.counts + priors, 10000) * mrr).sum(axis=1)
            b_results = (np.random.dirichlet(b_arm.counts + priors, 10000) * mrr).sum(axis=1)

            if b_results.mean() > a_results.mean() * 1.025:
                helper.not_bad()
                helper.not_good()
                return 2, None
            else:
                helper.not_bad()
                helper.not_good()
                # return None, None

        if a_arm.conversion_rate() > b_arm.conversion_rate() and testbed.get_p_value(a_arm.total_conversions(),
                                                                                     a_arm.total_samples(),
                                                                                     b_arm.total_conversions(),
                                                                                     b_arm.total_samples()) < 0.4:
            mrr = [5, 9, 30, 0]
            priors = np.array([1, 1, 1, 1])
            a_results = (np.random.dirichlet(a_arm.counts + priors, 10000) * mrr).sum(axis=1)
            b_results = (np.random.dirichlet(b_arm.counts + priors, 10000) * mrr).sum(axis=1)
            if b_results.mean() < a_results.mean():
                helper.not_bad()
                helper.not_good()
                return 1, None
            else:
                helper.not_bad()
                helper.not_good()
                # return None, None

        # After we hit a month, be more flexible don't care as much about payment numbers, care more about values.
        # If a>b stop the test. If b>a*1.25 b wins. if neither keep running through the possibilities
        if (a_arm.total_samples() + b_arm.total_samples()) > daily_views * 30:
            mrr = [5, 9, 30, 0]
            priors = np.array([1, 1, 1, 1])
            a_results = (np.random.dirichlet(a_arm.counts + priors, 10000) * mrr).sum(axis=1)
            b_results = (np.random.dirichlet(b_arm.counts + priors, 10000) * mrr).sum(axis=1)
            if b_results.mean() > a_results.mean() * 1.25:
                helper.not_bad()
                helper.not_good()
                return 2, None
            if b_results.mean() < a_results.mean():
                helper.not_bad()
                helper.not_good()
                return 1, None

    # Stop after two months
    if (a_arm.total_samples() + b_arm.total_samples()) > daily_views * 80:
        mrr = [5, 9, 30, 0]
        priors = np.array([1, 1, 1, 1])
        a_results = (np.random.dirichlet(a_arm.counts + priors, 10000) * mrr).sum(axis=1)
        b_results = (np.random.dirichlet(b_arm.counts + priors, 10000) * mrr).sum(axis=1)
        if b_results.mean() > a_results.mean() * 1.25:
            helper.not_bad()
            helper.not_good()
            return 2, None
        if b_results.mean() < a_results.mean():
            helper.not_bad()
            helper.not_good()
            return 1, None
        else:
            return 1, None

    helper.not_bad()
    helper.not_good()
    return None, None


def func3a(a_arm, b_arm):
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
                print('Secondary')
                return 2, None
            else:
                return 1, None
        else:
            return None, (1 - p_b_optimal)
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
        print('week')

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
# print(agg_results)
# print(_)
