import testbed1 as testbed
from math import sqrt
import numpy as np
from functools import partial


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


def tim_dynamic_threshold(a_arm, b_arm):
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
        if testbed.get_p_value(s_a, n_a, s_b, n_b) < (alpha / max_samples) * (n_a + n_b):

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


def dylans_rule(a_arm, b_arm):
    alpha = 0.05
    a_samples = a_arm.total_samples()
    b_samples = b_arm.total_samples()
    a_conversions = a_arm.total_conversions()
    b_conversions = b_arm.total_conversions()
    sample_size = 50000

    if (a_samples + a_samples >= sample_size):
        if (testbed.get_p_value(a_conversions, a_samples, b_conversions, b_samples) < alpha):
            if (a_arm.conversion_rate() > b_arm.conversion_rate()):
                return 1, None
            else:
                return 2, None
        else:
            return 1, None
    else:
        return None, None


def FacePuncher10o20(a_arm, b_arm):
    # Designed to drop small wins/losses quickly only want 5% wins or more

    if (a_arm.total_samples()+b_arm.total_samples())%100 ==0:
        def get_cash(multiplier = 1):
            mrr = [5, 9, 30, 0]
            priors = np.array([1, 1, 1, 1])
            a_results = (np.random.dirichlet(a_arm.counts + priors, 10000) * mrr).sum(axis=1)
            b_results = (np.random.dirichlet(b_arm.counts + priors, 10000) * mrr).sum(axis=1)
            if b_results.mean() > a_results.mean()*1.1:
                return '10+positive'
            if a_results.mean() > b_results.mean()*1.1:
                return '10+negative'
            if b_results.mean() > a_results.mean()*1.05:
                return '5+positive'
            if a_results.mean() > b_results.mean()*1.05:
                return '5+negative'
            if b_results.mean() > a_results.mean():
                return 'positive'
            if a_results.mean() > b_results.mean():
                return 'negative'

        if b_arm.total_conversions() >= 20 and get_cash() == '10+positive':
            return 2, None
        if a_arm.total_conversions() >= 20 and get_cash() == '10+negative':
            return 1, None
        if b_arm.total_conversions() >= 40 and get_cash() == '5+positive':
            return 2, None
        if a_arm.total_conversions() >= 40 and get_cash() == '5+negative':
            return 1, None
        if a_arm.total_conversions() >= 120 and get_cash() == 'positive':
            return 2, None
        else:
            return None, None
    return None, None


def FacePuncher10o25(a_arm, b_arm):
    # Designed to drop small wins/losses quickly only want 5% wins or more

    if (a_arm.total_samples()+b_arm.total_samples())%100 ==0:
        def get_cash(multiplier = 1):
            mrr = [5, 9, 30, 0]
            priors = np.array([1, 1, 1, 1])
            a_results = (np.random.dirichlet(a_arm.counts + priors, 10000) * mrr).sum(axis=1)
            b_results = (np.random.dirichlet(b_arm.counts + priors, 10000) * mrr).sum(axis=1)
            if b_results.mean() > a_results.mean()*1.1:
                return '10+positive'
            if a_results.mean() > b_results.mean()*1.1:
                return '10+negative'
            if b_results.mean() > a_results.mean()*1.05:
                return '5+positive'
            if a_results.mean() > b_results.mean()*1.05:
                return '5+negative'
            if b_results.mean() > a_results.mean():
                return 'positive'
            if a_results.mean() > b_results.mean():
                return 'negative'

        if b_arm.total_conversions() >= 25 and get_cash() == '10+positive':
            return 2, None
        if a_arm.total_conversions() >= 25 and get_cash() == '10+negative':
            return 1, None
        if b_arm.total_conversions() >= 50 and get_cash() == '5+positive':
            return 2, None
        if a_arm.total_conversions() >= 50 and get_cash() == '5+negative':
            return 1, None
        if a_arm.total_conversions() >= 150 and get_cash() == 'positive':
            return 2, None
        else:
            return None, None
    return None, None


def UtahRaptor(a_arm, b_arm):
    # After 100 conversions on B, check if it's 5% better, but if A ever becomes 10% worse, stop it.
    if b_arm.total_conversions() >= 100 and b_arm.conversion_rate() / a_arm.conversion_rate()-1 > 0.05:
        return 2, None
    if a_arm.total_conversions() > 1 and b_arm.total_conversions() > 1 and b_arm.conversion_rate() / a_arm.conversion_rate()-1 < 0.05:
        return 1, None
    return None, None


def Triceratops(a_arm, b_arm):
    # After 100 conversions on B, check if it's 5% better, but if A ever becomes 10% worse, stop it.
    if b_arm.total_conversions() >= 100 and b_arm.conversion_rate() / a_arm.conversion_rate()-1 > 0.10:
        return 2, None
    if a_arm.total_conversions() > 1 and b_arm.total_conversions() > 1 and b_arm.conversion_rate() / a_arm.conversion_rate()-1 < 0.05:
        return 1, None
    return None, None


def buttered_toast(a_arm, b_arm, start_certainty=.8, end_certainty=.6, count=5000, final_certainty=.90):
    if (sum(a_arm.counts) + sum(b_arm.counts)) % 100 == 0:
        mrr = [5, 9, 30, 0]
        priors = np.array([1, 1, 1, 1])
        a_results = (np.random.dirichlet(a_arm.counts + priors, 10000) * mrr).sum(axis=1)
        b_results = (np.random.dirichlet(b_arm.counts + priors, 10000) * mrr).sum(axis=1)

        # add a rolling certainty requirement
        m_certainty = start_certainty - ((start_certainty-end_certainty)/count * (sum(a_arm.counts) + sum(b_arm.counts)))

        if sum(a_results < b_results) / len(a_results) > m_certainty:
            return 2, None
        elif sum(b_results < a_results) / len(a_results) > m_certainty:
            return 1, None
        elif sum(b_arm.counts) >= count/2:
            if sum(a_results < b_results) / len(a_results) > final_certainty:
                return 2, None
            else:
                return 1, None
        else:
            return None, .15
    else:
        return None, None


def english_muffin(a_arm, b_arm, certainty=.70, count=4500, final_certainty=.90):
    if (sum(a_arm.counts) + sum(b_arm.counts)) % 500 == 0:
        mrr = [5, 9, 30, 0]
        priors = np.array([1, 1, 1, 1])
        a_results = (np.random.dirichlet(a_arm.counts + priors, 10000) * mrr).sum(axis=1)
        b_results = (np.random.dirichlet(b_arm.counts + priors, 10000) * mrr).sum(axis=1)

        if sum(a_results < b_results) / len(a_results) > certainty:
            return 2, None
        elif sum(b_results < a_results) / len(a_results) > certainty:
            return 1, None
        elif sum(b_arm.counts) >= count/2:
            if sum(a_results < b_results) / len(a_results) > final_certainty:
                return 2, None
            else:
                return 1, None
        else:
            return None, .15
    else:
        return None, None


def one_rule_to_rule_them_all(a, b):
    mrr = np.array([5, 9, 30, 0])

    if a.total_samples() < 50 or b.total_samples() < 50:
        return None, None
    elif a.total_samples() + b.total_samples() > 6000:
        return (1 if (sum(a.counts * mrr) / a.total_samples() > sum(b.counts * mrr) / b.total_samples()) else 2), None
    else:
        aExpected = 0
        for x in range(0, 3):
            aExpected += beta.rvs(a.counts[x] + 1, 1 + a.total_samples() - a.counts[x]) * mrr[x]
        bExpected = 0
        for x in range(0, 3):
            bExpected += beta.rvs(b.counts[x] + 1, 1 + b.total_samples() - b.counts[x]) * mrr[x]
        return None, (1 if aExpected > bExpected else 0)


if __name__ == '__main__':
    my_helper = SpencerHelper()
    test_partial = partial(spencer_rule, helper=my_helper)
    testbed.multi_test([test_partial, tim_dynamic_threshold, tim_thompson_sampling, dylans_rule, FacePuncher10o20, FacePuncher10o25, UtahRaptor, Triceratops, buttered_toast, english_muffin, one_rule_to_rule_them_all], max_tests=9999, plot=True,
                       seed=np.random.random())
