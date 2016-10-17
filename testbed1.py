# A/B simulator testbed
import numpy as np
import pandas as pd
from math import sqrt, ceil, floor, log

import time
from scipy.stats import norm, beta
from matplotlib import pyplot as plt
from multiprocessing import Queue, Process
import os


# This is a test arm class that lets me deal with test arms in a better way
class TestArm:
    def __init__(self):
        self.counts = [0, 0, 0, 0]

    def total_conversions(self):
        return sum(self.counts[:3])

    def total_samples(self):
        return sum(self.counts)

    def conversion_rate(self):
        return self.total_conversions() / self.total_samples()

    def update(self, level):
        self.counts[level] += 1


def get_p_value(a_success, a_total, b_success, b_total, alternative='two sided'):
    if a_total == 0 or b_total == 0:
        return None

    p_a_temp = a_success / a_total
    p_b_temp = b_success / b_total

    # calculate proportions and significance
    p_pooled = (a_success + b_success) / (a_total + b_total)
    standard_error = sqrt(p_pooled * (1 - p_pooled) * (1 / a_total + 1 / b_total))
    if standard_error == 0:
        return 1
    z = (p_a_temp - p_b_temp) / standard_error

    if alternative == 'two sided':
        p_value = 2 * norm.pdf(-abs(z))
    elif alternative == 'greater':
        p_value = norm.pdf(z)
    elif alternative == 'less':
        p_value = norm.pdf(-z)

    return p_value


# returns the probability that b is the optimal arm in terms of expected value per user
def get_p_b_optimal(a_arm, b_arm, priors=[1, 1, 1, 1], mrr=[5, 9, 30, 0], n=10000):
    priors = np.array(priors)
    a_results = (np.random.dirichlet(a_arm.counts + priors, n) * mrr).sum(axis=1)
    b_results = (np.random.dirichlet(b_arm.counts + priors, n) * mrr).sum(axis=1)

    return sum(a_results < b_results) / len(a_results)


def run_test(stopping_rule, q, plot_q, mrr=[5, 9, 30, 0], n=10000, p_baseline_default=[.010, .0082, .0025],
             max_tests=10000, max_people=1000000, test_size=50000, m_axis=None, seed=False):
    try:
        stopping_rule.__name__
    except AttributeError:
        stopping_rule.__name__ = 'Unnamed_rule'

    # use a seed for the competition
    if seed is not False:
        np.random.seed(seed)

    p_baseline = p_baseline_default + [1 - sum(p_baseline_default)]

    # Helper Functions
    def get_sample(prop_a=0.5):
        sample_result = np.argmax(
            np.random.multinomial(1, np.concatenate([prop_a * np.array(p_a), (1 - prop_a) * np.array(p_b)])))
        if sample_result < 4:
            a_arm.update(sample_result)
        else:
            b_arm.update(sample_result - 4)

    def ev(arm_p):
        return (np.array(arm_p) * mrr).sum()

    def calculate_revenue(arm):
        return (np.array(arm.counts) * mrr).sum()

    prob_a = np.repeat([p_baseline], n, axis=0)

    # Choose test parameters using a copula

    # Control the probability of B winning

    def get_copula(shift=.98, scale=9000, uncertainty_ind=.8, uncertainty_team=.6, baseline=[.010, .0082, .0025],
                   cop_n=10000):

        corr_matrix = np.array([[1, 0.89528368 * uncertainty_ind, 0.824624949 * uncertainty_team],
                                [0.89528368 * uncertainty_ind, 1, 0.831348201 * uncertainty_team],
                                [0.824624949 * uncertainty_team, 0.831348201 * uncertainty_team, 1]])
        mean = [0, 0, 0]
        z = np.random.multivariate_normal(mean=mean, cov=corr_matrix, size=cop_n)
        y = norm.cdf(z)

        x1 = beta(baseline[0] * shift * scale, (1 - baseline[0] * shift) * scale).ppf(y[:, 0])
        x2 = beta(baseline[1] * shift * scale, (1 - baseline[1] * shift) * scale).ppf(y[:, 1])
        x3 = beta(baseline[2] * shift * scale, (1 - baseline[2] * shift) * scale).ppf(y[:, 2])

        m_dist = np.vstack((x1, x2, x3,)).T
        m_dist = np.hstack((m_dist, 1 - m_dist.sum(axis=1).reshape(len(m_dist), 1)))
        return m_dist

    prob_b = get_copula()

    p_a = prob_a[0]
    p_b = prob_b[0]

    a_arm = TestArm()
    b_arm = TestArm()

    test_size_counter = 0
    people_count = 0
    test_count = 1
    proportion_a = .5
    proportion_a_prev = .5
    results = []

    while test_count <= max_tests and people_count <= max_people:  # Added max-people to the count

        # Get a new sample and increment the people count
        get_sample(proportion_a)
        people_count += 1

        # This counter is a fast way of keeping track of how many people have been in each test
        test_size_counter += 1

        # Pass the results of the test (tests must accept two lists of len = 4 and return 1, 2, or None)
        # Have the second returned var be T-A Prop.
        choice, proportion_a = stopping_rule(a_arm, b_arm)
        if proportion_a is None:
            proportion_a = proportion_a_prev
        else:
            proportion_a_prev = proportion_a

        if test_size_counter >= test_size:
            choice = 3

        if choice:
            if choice == 3:
                m_choice = 'No Choice'
            elif choice == 2:
                m_choice = 'B'
            else:
                m_choice = 'A'

            # Calculate some per-test values
            test_value = calculate_revenue(a_arm) + calculate_revenue(b_arm) + (ev(p_a) if choice == 1 else ev(
                p_b)) * (test_size - test_size_counter)
            optimal_test_value = ev(p_a) * test_size if ev(p_a) > ev(p_b) else ev(p_b) * test_size
            estimated_revenue = calculate_revenue(a_arm) + calculate_revenue(b_arm) + (
                ((calculate_revenue(a_arm) / a_arm.total_samples()) * (
                test_size - test_size_counter)) if choice == 1 else (
                    (calculate_revenue(b_arm) / b_arm.total_samples()) * (test_size - test_size_counter)))

            temp_results = {'Test Name': stopping_rule.__name__, 'Test Number': test_count, 'A Basic': p_a[0],
                            ' A Pro': p_a[1],
                            'A Team': p_a[2], 'A Free': p_a[3], 'B Basic': p_b[0], ' B Pro': p_b[1],
                            'B Team': p_b[2], 'B Free': p_b[3], 'EV A': ev(p_a), 'EV B': ev(p_b),
                            'A Success': a_arm.total_conversions(), 'A Number': a_arm.total_samples(),
                            'B Success': b_arm.total_conversions(),
                            'B Number': b_arm.total_samples(),
                            'Total Number': a_arm.total_samples() + b_arm.total_samples(),
                            'Choice': m_choice, 'Actual Winner': 'A' if ev(p_a) > ev(p_b) else 'B',
                            'A Revenue': calculate_revenue(a_arm), 'B Revenue': calculate_revenue(b_arm),
                            'Regret': (ev(p_a) - ev(p_b)) * b_arm.total_samples() if ev(p_a) > ev(p_b) else (ev(
                                p_b) - ev(
                                p_a)) * a_arm.total_samples(),
                            'EV A Measured': calculate_revenue(a_arm) / a_arm.total_samples(),
                            'EV B Measured': calculate_revenue(b_arm) / b_arm.total_samples(),
                            'True Revenue': test_value, 'Estimated Revenue': estimated_revenue,
                            'Optimal Value': optimal_test_value}

            results.append(temp_results)  # This is now a list of dicts that can easily be appended to a DataFrame

            if m_axis is not None:
                if temp_results['Choice'] == 'A' and temp_results['Actual Winner'] == 'B':
                    m_color = 'orange'
                elif temp_results['Choice'] == 'B' and temp_results['Actual Winner'] == 'A':
                    m_color = 'red'
                elif temp_results['Choice'] == 'A' and temp_results['Actual Winner'] == 'A':
                    m_color = 'blue'
                elif temp_results['Choice'] == 'B' and temp_results['Actual Winner'] == 'B':
                    m_color = 'darkgreen'
                else:
                    m_color = 'black'

                x = temp_results['EV B'] - temp_results['EV A']
                y = temp_results['EV B Measured'] - temp_results['EV A Measured']
                color = m_color
                size = (temp_results['A Number'] + temp_results['B Number']) / 750
                plot_q.put((x, y, m_axis, color, size))

            print(stopping_rule.__name__ + ' finished test #' + str(test_count) + ': ' + str(
                people_count) + ' people tested so far')

            # Increment test count
            test_count += 1

            # Reset test_size_counter
            test_size_counter = 0

            # Generate a new Test
            p_a = prob_a[test_count]
            p_b = prob_b[test_count]

            # Start new test arms
            a_arm = TestArm()
            b_arm = TestArm()

            # Reset the proportions for the arms
            proportion_a = 0.5
            proportion_a_prev = 0.5

    optimal_rule_value = np.maximum((prob_b[:max_tests] * mrr).sum(axis=1), (prob_a[0] * mrr).sum()).sum() * test_size

    q.put((results, stopping_rule.__name__, optimal_rule_value))
    plot_q.put(('finished',))


def multi_test(decision_rules, mrr=[5, 9, 30, 0], n=10000, p_baseline=[.010, .0082, .0025], max_tests=1000,
               max_people=10000, test_size=50000, plot=True, seed=False):
    q = Queue()
    plot_q = Queue()

    # Leave these in for Column Order
    ind_test_results = pd.DataFrame(
        columns=['Test Name', 'Test Number', 'A Basic', ' A Pro', 'A Team', 'A Free', 'B Basic', ' B Pro',
                 'B Team', 'B Free', 'EV A', 'EV B', 'A Success', 'A Number', 'B Success',
                 'B Number', 'Total Number', 'Choice', 'Actual Winner', 'A Revenue', 'B Revenue',
                 'Regret', 'EV A Measured', 'EV B Measured', 'True Revenue', 'Estimated Revenue', 'Optimal Value'])
    agg_test_results = pd.DataFrame(
        columns=['Test Name', 'Test Count', 'People Count', 'True Positive', 'False Positive',
                 'True Negative', 'False Negative', 'True Positive Rate',
                 'True Negative Rate', 'Positive Predictive Value',
                 'Negative Predictive Value', 'Regret', 'Revenue', 'True Revenue', 'Estimated Revenue',
                 'Optimal Value'])

    axes_count = 0
    m_procs = []
    for rule in decision_rules:
        if plot:
            axis_num = axes_count
        else:
            axis_num = None
        # Actually Run the tests
        m_procs.append(Process(target=run_test, args=(
        rule, q, plot_q, mrr, n, p_baseline, max_tests, max_people, test_size, axis_num, seed)))

        if plot:
            axes_count += 1

    for p in m_procs:
        p.start()

    if plot:

        def get_axis(m_axis_num):
            if len(m_procs) == 1:
                return axes
            elif len(m_procs) <= 3:
                return axes[m_axis_num]
            else:
                return axes[floor(m_axis_num / m_size)][m_axis_num % m_size]

        plt.ion()

        # set up figure with correct dimensions
        if len(decision_rules) > 3:
            m_size = ceil(sqrt(len(decision_rules)))
            multi_plt, axes = plt.subplots(m_size, m_size)
        else:
            multi_plt, axes = plt.subplots(len(decision_rules))

        # Add labels to plots
        label_count = 0
        for rule in decision_rules:
            # Check for a problem with finding the stopping rule name
            try:
                rule.__name__
            except AttributeError:
                rule.__name__ = 'Unnamed_rule'

            m_axis = get_axis(label_count)
            m_axis.set_title(rule.__name__)
            m_axis.set_xlabel('True Difference in EV')
            m_axis.set_ylabel('Measured Difference in EV')
            m_axis.set_xlim([-.2, .2])
            m_axis.set_ylim([-.5, .5])
            label_count += 1

        finished_count = 0
        point_counter = 0
        while finished_count < len(m_procs):

            val = plot_q.get()
            if val[0] == 'finished':
                finished_count += 1
            else:
                x, y, axis, color, dot_size = val

                get_axis(axis).scatter(x, y, color=color, s=dot_size)

                get_axis(axis).figure.canvas.draw()
                plt.pause(1e-8)

            point_counter += 1
            if point_counter % 50 == 0:
                print(str(point_counter) + ' points plotted')

        filename = 'results/comparison/' + time.strftime('%Y%m%d_%H-%M_') + 'Agg Results.png'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        multi_plt.savefig(filename)

    for p in m_procs:
        test_data, test_name, optimal_value = q.get()
        test_result = pd.DataFrame(test_data, columns=list(ind_test_results.columns))
        ind_test_results = pd.concat([ind_test_results, test_result], ignore_index=True)

        true_positive = test_result[(test_result['Choice'] == 'B') & (test_result['Actual Winner'] == 'B')]
        false_positive = test_result[(test_result['Choice'] == 'B') & (test_result['Actual Winner'] == 'A')]
        true_negative = test_result[(test_result['Choice'] == 'A') & (test_result['Actual Winner'] == 'A')]
        false_negative = test_result[(test_result['Choice'] == 'A') & (test_result['Actual Winner'] == 'B')]

        temp_agg = [
            {'Test Name': test_name, 'Test Count': len(test_result), 'People Count': test_result['Total Number'].sum(),
             'True Positive': len(true_positive),
             'False Positive': len(false_positive),
             'True Negative': len(true_negative), 'False Negative': len(false_negative),
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
             'Regret': test_result.Regret.sum() / test_result['Total Number'].sum() * 1000000,
             'Revenue': test_result[['A Revenue', 'B Revenue']].sum().sum() / test_result[
                 'Total Number'].sum() * 1000000,
             'True Revenue': test_result['True Revenue'].sum(),
             'Estimated Revenue': test_result['Estimated Revenue'].sum(),
             'Optimal Value': optimal_value
             }]
        temp_agg = pd.DataFrame(temp_agg, columns=list(agg_test_results.columns))
        filename = 'results/' + test_name + '/' + time.strftime('%Y%m%d_%H-%M_') + test_name + ".csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        temp_agg.to_csv(filename, index=False)
        agg_test_results = pd.concat([agg_test_results, temp_agg], ignore_index=True)
        # Plot Stuff for each test
        if plot:
            ind_figure = plt.figure(3)
            ind_figure.add_subplot(111)
            ind_figure.axes[0].set_xlabel("True Difference in EV")
            ind_figure.axes[0].set_ylabel('Measured Difference in EV')
            ind_figure.axes[0].set_title(test_name)
            ind_figure.axes[0].scatter(false_negative['EV B'] - false_negative['EV A'],
                                       false_negative['EV B Measured'] - false_negative['EV A Measured'],
                                       color='orange')
            ind_figure.axes[0].scatter(true_positive['EV B'] - true_positive['EV A'],
                                       true_positive['EV B Measured'] - true_positive['EV A Measured'],
                                       color='darkgreen')
            ind_figure.axes[0].scatter(true_negative['EV B'] - true_negative['EV A'],
                                       true_negative['EV B Measured'] - true_negative['EV A Measured'], color='blue')
            ind_figure.axes[0].scatter(false_positive['EV B'] - false_positive['EV A'],
                                       false_positive['EV B Measured'] - false_positive['EV A Measured'], color='red')
            filename = 'results/' + test_name + '/' + time.strftime('%Y%m%d_%H-%M_') + test_name + ".png"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            ind_figure.savefig(filename)
            plt.close(ind_figure)

    for p in m_procs:
        p.join()
        filename = 'results/comparison/' + time.strftime('%Y%m%d_%H-%M_') + "Agg Results.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        agg_test_results.to_csv('results/comparison/' + time.strftime('%Y%m%d_%H-%M_') + 'Agg Results.csv', index=False)
        ind_test_results.to_csv('results/comparison/' + time.strftime('%Y%m%d_%H-%M_') + 'Ind Results.csv', index=False)

    return agg_test_results, ind_test_results
