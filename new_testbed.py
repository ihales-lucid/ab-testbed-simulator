import numpy as np
import pandas as pd
from scipy.stats import norm, beta
from multiprocessing import Queue, Process
import os
import time

mrr = np.array([5, 9, 30, 0])


# Test Class
class ABTest:
    def __init__(self, tb_raw):
        self.counts = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
        self.tb_raw = tb_raw
        self.prop_a = .5
        self.payload = None

    def total_conversions(self):
        return self.counts[:, :3].sum(axis=1)

    def total_samples(self):
        return self.counts.sum(axis=1)

    def conversion_rate(self):
        return self.total_conversions() / self.total_samples()

    def update(self, arm, level):
        self.counts[arm][level] += 1

    def finalize_test(self, result, count, original_baseline, new_baseline, tb_scaled):
        # Do finalize stuff here
        m_results = {'Test Count': count,
                     'Test Name': test_rule.__name__,
                     'Samples': self.counts.sum(),
                     'Best Arm': 'B' if (((tb_scaled * original_baseline) * mrr[:3]).sum() > (original_baseline * mrr[:3]).sum() - 1) else 'A',
                     'Choice': 'A' if result == 0 else 'B',
                     'Observed Difference': (self.counts[1] * mrr).sum() / (self.counts[0] * mrr).sum() - 1,
                     'Original Baseline Value': (original_baseline * mrr[:3]).sum(),
                     'New Baseline Value': (new_baseline * mrr[:3]).sum(),
                     'Test-Time Revenue': (self.counts * mrr).sum(),
                     #                     'T-A Value': (original_baseline * mrr[:3]).sum(),
                     #                     'T-B Value': (tb_scaled * original_baseline * mrr[:3]).sum()
                     }
        return m_results

    def set_prop_a(self, prop_a):
        self.prop_a = prop_a

    def get_assignment(self):
        return [0, 0, 0] if np.random.binomial(1, 1 - self.prop_a) == 0 else self.tb_raw

    def get_payload(self):
        return self.payload

    def set_payload(self, x):
        self.payload = x


# Run Tests
def run_test(rule, max_people, max_concurrent, cadence, seed, q):
    """ This function runs tests against the rule in 'rule' parameter. Max people is the total number of people to test.
    max_concurrent is the total number of tests that can run concurrently. Cadence is how many people need to be tested
    before a new batch of tests can start. Seed is the random seed for the test generation. """

    def add_tests(test_array):
        # Reference the T-B Counter (a pointer for tb_vector access
        nonlocal tb_counter
        for _ in range(max_concurrent - len(test_array)):
            test_array.append(ABTest(tb_vector[tb_counter]))
            tb_counter += 1
        return test_array

    # Get T-B Vector - Difference from baseline
    def get_copula(shift=.98, scale=9000, uncertainty_ind=.8, uncertainty_team=.6, m_baseline=[.010, .0082, .0025],
                   cop_n=50000):
        np.random.seed(seed)  # Add a seed for random number gen.
        corr_matrix = np.array([[1, 0.89528368 * uncertainty_ind, 0.824624949 * uncertainty_team],
                                [0.89528368 * uncertainty_ind, 1, 0.831348201 * uncertainty_team],
                                [0.824624949 * uncertainty_team, 0.831348201 * uncertainty_team, 1]])
        mean = [0, 0, 0]
        z = np.random.multivariate_normal(mean=mean, cov=corr_matrix, size=cop_n)
        y = norm.cdf(z)
        x1 = beta(m_baseline[0] * shift * scale, (1 - m_baseline[0] * shift) * scale).ppf(y[:, 0])
        x2 = beta(m_baseline[1] * shift * scale, (1 - m_baseline[1] * shift) * scale).ppf(y[:, 1])
        x3 = beta(m_baseline[2] * shift * scale, (1 - m_baseline[2] * shift) * scale).ppf(y[:, 2])
        m_dist = np.vstack((x1, x2, x3,)).T
        # m_dist = np.hstack((m_dist, 1 - m_dist.sum(axis=1).reshape(len(m_dist), 1)))
        return m_dist

    # Do what we need to do to scale the TB data
    def scale_tb(m_raw):
        #        return np.array(tb_raw) * get_scale_factor() + 1
        return np.array(m_raw) + 1

    # Calculate the "scale factor"
    def get_scale_factor(max_difference=np.array([1, 1, 1])):

        scale_factor = (max_difference - baseline) / (max_difference - original_baseline)
        for j, x in enumerate(scale_factor):
            if x > 1:
                scale_factor[j] = 1
            elif x < 0:
                scale_factor[j] = 0

        return scale_factor

    # Initiate Baseline Conversion Rates
    baseline = [.010, .0082, .0025]
    original_baseline = baseline.copy()
    # baseline.append(1 - sum(baseline))

    tb_vector = get_copula()
    tb_vector = tb_vector / baseline - 1
    tb_counter = 0

    # Initialize max concurrent tests
    m_tests = []
    m_tests = add_tests(m_tests)

    # Initialize the people and test counters
    people_count = 0
    test_count = 0

    # Results collector
    m_results = []

    # At this point I should have a list of tests that will run at the same time. Now we need to run the rule on them
    # Start the main testing loop
    while people_count <= max_people:
        if people_count % cadence == 0:
            add_tests(m_tests)
        # Start a new person
        people_count += 1

        # Get test assignment sand apply a scale factor to the T-B assignments
        test_assignments = np.array([m_test.get_assignment() for m_test in m_tests])
        test_probabilities = np.array(
            [baseline if np.array_equal(x, [0, 0, 0]) else scale_tb(x) for x in test_assignments])

        # Get the vector of combined probabilities for the person
        combined_probability = test_probabilities.prod(axis=int(0)) * baseline

        # Add the probability of Free
        combined_probability = np.append(combined_probability, 1 - np.array(combined_probability).sum())

        # Choose the winner
        sample_result = np.argmax(np.random.multinomial(1, combined_probability))

        # Update each test with the winner counts
        for i, x in enumerate(m_tests):
            x.update(0 if np.array_equal(test_assignments[i], [0, 0, 0]) else 1, sample_result)

        # We've now chosen the winner and updated the tests to reflect the winner. Now run the test arms on the winners.
        # del_list is a list of tests to drop (because they've been called

        del_list = []
        for i, x in enumerate(m_tests):
            result = rule(x)
            if result is not None:
                test_count += 1
                new_baseline = baseline * (
                    1 if np.array_equal(test_assignments[i], [0, 0, 0]) else scale_tb(test_assignments[i]))
                m_results.append(
                    x.finalize_test(result, test_count, baseline, new_baseline, scale_tb(x.tb_raw)))
                del_list.append(i)
                # Update the baseline based on the scaled value of this test
                baseline = new_baseline
                print('Test #' + str(test_count) + ' completed. ' + str(people_count) + ' people tested so far.')

        # Drop called tests
        if len(del_list) > 0:
            del_list.sort(reverse=True)
            for i in del_list:
                del m_tests[i]

    q.put(m_results)


def multi_test(rules, max_people, max_concurrent, cadence, seed=None):
    q = Queue()

    ind_results = pd.DataFrame()
    agg_results = pd.DataFrame()

    # Create a new process for each stopping rule
    m_procs = []
    for rule in rules:
        m_procs.append(Process(target=run_test, args=(rule, max_people, max_concurrent, cadence, seed, q)))

    # Start all stopping rule processes
    for p in m_procs:
        p.start()

    for p in m_procs:
        test_results = q.get()  # This gets the list of dicts of individual test results ({column: value})
        temp_results = pd.DataFrame(test_results)
        ind_results = pd.concat([ind_results, temp_results], ignore_index=True)  # Add the individual test results

        # Now lets get the AGG data
        true_positive = temp_results[(temp_results['Choice'] == 'B') & (temp_results['Best Arm'] == 'B')]
        false_positive = temp_results[(temp_results['Choice'] == 'B') & (temp_results['Best Arm'] == 'A')]
        true_negative = temp_results[(temp_results['Choice'] == 'A') & (temp_results['Best Arm'] == 'A')]
        false_negative = temp_results[(temp_results['Choice'] == 'A') & (temp_results['Best Arm'] == 'B')]

        m_revenue = temp_results['Test-Time Revenue'].sum()

        temp_agg = [
            {
                'Test Name': temp_results['Test Name'].values[0],
                'Test Count': len(temp_results),
                'People Count': temp_results.Samples.sum(),
                'True Positive': len(true_positive),
                'False Positive': len(false_positive),
                'True Negative': len(true_negative),
                'False Negative': len(false_negative),
                'True Positive Rate': (len(true_positive) / (len(true_positive) + len(false_negative)) if (
                    len(true_positive) + len(false_negative) != 0) else 0),
                'True Negative Rate': (len(true_negative) / (len(true_negative) + len(false_positive)) if (
                    len(true_negative) + len(false_positive) != 0) else 0),
                'Positive Predictive Value': (
                    len(true_positive) / (len(true_positive) + len(false_positive)) if (len(true_positive) + len(
                        false_positive)) != 0 else 0),
                'Negative Predictive Value': (
                    len(true_negative) / (len(true_negative) + len(false_negative)) if (len(true_negative) + len(
                        false_negative)) != 0 else 0),
                'Revenue': m_revenue,
                'Final Per-User Value': temp_results['New Baseline Value'].values[-1]
            }
        ]

        temp_agg = pd.DataFrame(temp_agg)

        agg_results = pd.concat([agg_results, temp_agg], ignore_index=True)  # Append the aggregate measure to DF

    # Now let's save the aggregate data
    for p in m_procs:
        p.join()
        filename = 'results/test_comparison/' + time.strftime('%Y%m%d_%H-%M_') + "Agg Results.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        agg_results.to_csv('results/test_comparison/' + time.strftime('%Y%m%d_%H-%M_') + 'Agg Results.csv', index=False)
        ind_results.to_csv('results/test_comparison/' + time.strftime('%Y%m%d_%H-%M_') + 'Ind Results.csv', index=False)


def test_rule(m_test):
    if m_test.counts.sum() >= 5:
        return np.random.binomial(1, .2)
    else:
        return None


if __name__ == '__main__':
    multi_test([test_rule], 300, 1, 10)
