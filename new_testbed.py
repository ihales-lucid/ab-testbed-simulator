import numpy as np
from scipy.stats import norm, beta


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

    def finalize_test(self, result):
        # Do finalize stuff here
        pass

    def set_prop_a(self, prop_a):
        self.prop_a = prop_a

    def get_assignment(self):
        return [0, 0, 0] if np.random.binomial(1, 1 - self.prop_a) == 0 else self.tb_raw

    def get_payload(self):
        return self.payload

    def set_payload(self, x):
        self.payload = x


# Run Tests
def run_test(rule, max_people, max_concurrent, cadence):
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
    def scale_tb(tb_raw):
        return np.array(tb_raw) * get_scale_factor() + 1

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
            print('cadence')
            add_tests(m_tests)

        # Start a new person
        people_count += 1

        # Get test assignment sand apply a scale factor to the T-B assignments
        test_assignments = np.array([m_test.get_assignment() for m_test in m_tests])
        test_probabilities = np.array(
            [([1, 1, 1]) if np.array_equal(x, [0, 0, 0]) else scale_tb(x) for x in test_assignments])

        # Get the vector of combined probabilities for the person
        combined_probability = test_probabilities.prod(axis=int(0)) * baseline

        # Add the probability of Free
        np.append(combined_probability, 1 - np.array(combined_probability).sum())

        # Choose the winner
        sample_result = np.argmax(np.random.multinomial(1, combined_probability))

        # Update each test with the winner counts
        for i, x in enumerate(m_tests):
            x.update(0 if np.array_equal(test_assignments[i], [0, 0, 0, 0]) else 1, sample_result)
            i += 1

        # We've now chosen the winner and updated the tests to reflect the winner. Now run the test arms on the winners.

        for i, x in enumerate(m_tests):
            result = rule(x)
            if result is not None:
                test_count += 1
                m_results.append(x.finalize_test(result, test_count))
                m_tests.remove(x)

                # Update the baseline based on the scaled value of this test
                baseline *= (
                    1 if np.array_equal(test_assignments[i], [0, 0, 0]) else scale_tb(test_assignments[i]))
                print('Test #' + str(test_count) + ' completed. ' + str(people_count) + ' people tested so far.')
                print(baseline, original_baseline)

    print(test_count, people_count)
    print(m_results)


def test_rule(m_test):
    if m_test.total_samples().sum() >= 1000:
        return np.random.binomial(1, .2)
    else:
        return None


run_test(test_rule, 1000000, 2, 4000)
