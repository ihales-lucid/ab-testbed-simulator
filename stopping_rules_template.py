import testbed1 as testbed

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


def example_rule(a_arm, b_arm):
    """This is an exceptionally bad stopping rule, but it should give you an idea"""

    if a_arm.total_samples() + b_arm.total_samples() > 1000:
        p_value = testbed.get_p_value(a_arm.total_conversions(), a_arm.total_samples(), b_arm.total_conversions(),
                                      b_arm.total_samples())
        if p_value < .05:
            return 2, None
        elif p_value > .95:
            return 1, None
        elif a_arm.total_samples() + b_arm.total_samples() > 50000:
            return 1, None
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

    testbed.multi_test([example_rule, example_rule], max_tests=10, plot=True, seed=True)
