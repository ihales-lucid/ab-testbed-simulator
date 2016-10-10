# A/B simulator testbed
import numpy as np
import pandas as pd
from math import sqrt, ceil, floor

import time
from scipy.stats import norm
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
        return None
    z = (p_a_temp - p_b_temp) / standard_error

    if alternative == 'two sided':
        p_value = 2 * norm.pdf(-abs(z))
    elif alternative == 'greater':
        p_value = norm.pdf(z)
    elif alternative == 'less':
        p_value = norm.pdf(-z)

    return p_value

# returns the probability that b is the optimal arm in terms of expected value per user
def get_p_b_optimal(a_arm, b_arm, a_priors = [1, 1, 1, 1], b_priors = [1, 1, 1, 1], mrr = [5, 9, 30, 0], n = 10000):
    a_priors = np.array(a_priors)
    b_priors = np.array(b_priors)
    a_results = (np.random.dirichlet(a_arm.counts + a_priors, n) * mrr).sum(axis=1)
    b_results = (np.random.dirichlet(b_arm.counts + b_priors, n) * mrr).sum(axis=1)

    return sum(a_results < b_results) / len(a_results)


def run_test(stopping_rule, q, plot_q, mrr=[5, 9, 30, 0], n=10000, p_baseline_default=[.010, .0082, .0025],
             max_tests=10000, m_axis=None, seed=False):
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

    # Choose test parameters from dirichlet distribution

    # Control the probability of B winning
    d_scale_factor = 10000
    d_shift_factor = .95
    prob_b = np.random.dirichlet(d_scale_factor * np.array(p_baseline) * ([d_shift_factor] * 3 + [1]), n)

    p_a = prob_a[0]
    p_b = prob_b[0]

    a_arm = TestArm()
    b_arm = TestArm()

    people_count = 0
    test_count = 1
    proportion_a = .5
    proportion_a_prev = .5
    results = []


    while test_count <= max_tests:

        # Get a new sample and increment the people count
        get_sample(proportion_a)
        people_count += 1

        # Pass the results of the test (tests must accept two lists of len = 4 and return 1, 2, or None)
        # eventually have the second return var be T-A Prop.
        choice, proportion_a = stopping_rule(a_arm, b_arm)
        if proportion_a is None:
            proportion_a = proportion_a_prev
        else:
            proportion_a_prev = proportion_a

        if choice:
            temp_results = {'Test Name': stopping_rule.__name__, 'Test Number': test_count, 'A Basic': p_a[0],
                            ' A Pro': p_a[1],
                            'A Team': p_a[2], 'A Free': p_a[3], 'B Basic': p_b[0], ' B Pro': p_b[1],
                            'B Team': p_b[2], 'B Free': p_b[3], 'EV A': ev(p_a), 'EV B': ev(p_b),
                            'A Success': a_arm.total_conversions(), 'A Number': a_arm.total_samples(),
                            'B Success': b_arm.total_conversions(),
                            'B Number': b_arm.total_samples(),
                            'Total Number': a_arm.total_samples() + b_arm.total_samples(),
                            'Choice': 'A' if choice == 1 else 'B', 'Actual Winner': 'A' if ev(p_a) > ev(p_b) else 'B',
                            'A Revenue': calculate_revenue(a_arm), 'B Revenue': calculate_revenue(b_arm),
                            'Regret': (ev(p_a) - ev(p_b)) * b_arm.total_samples() if ev(p_a) > ev(p_b) else (ev(
                                p_b) - ev(
                                p_a)) * a_arm.total_samples(),
                            'EV A Measured': calculate_revenue(a_arm) / a_arm.total_samples(),
                            'EV B Measured': calculate_revenue(b_arm) / b_arm.total_samples(),
                            'EV True Incremental': ev(p_b) - ev(p_a) if choice == 2 else 0,
                            'EV Measured Incremental': calculate_revenue(
                                b_arm) / b_arm.total_samples() - calculate_revenue(
                                a_arm) / a_arm.total_samples() if choice == 2 else 0}

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

                x = temp_results['EV B'] - temp_results['EV A']
                y = temp_results['EV B Measured'] - temp_results['EV A Measured']
                color = m_color
                plot_q.put((x, y, m_axis, color))

            print(stopping_rule.__name__ + ' finished test #' + str(test_count) + ': ' + str(
                people_count) + ' people tested so far')

            # Increment test count
            test_count += 1

            # Generate a new Test
            p_a = prob_a[test_count]
            p_b = prob_b[test_count]

            # Start new test arms
            a_arm = TestArm()
            b_arm = TestArm()

            # Reset the proportions for the arms
            proportion_a = 0.5
            proportion_a_prev = 0.5

    q.put((results, stopping_rule.__name__))
    plot_q.put(('finished',))


def multi_test(decision_rules, mrr=[5, 9, 30, 0], n=10000, p_baseline=[.010, .0082, .0025], max_tests=1000,
               plot=True, seed=False):
    q = Queue()
    plot_q = Queue()

    ind_test_results = pd.DataFrame(
        columns=['Test Name', 'Test Number', 'A Basic', ' A Pro', 'A Team', 'A Free', 'B Basic', ' B Pro',
                 'B Team', 'B Free', 'EV A', 'EV B', 'A Success', 'A Number', 'B Success',
                 'B Number', 'Total Number', 'Choice', 'Actual Winner', 'A Revenue', 'B Revenue',
                 'Regret', 'EV A Measured', 'EV B Measured', 'EV True Incremental',
                 'EV Measured Incremental'])
    agg_test_results = pd.DataFrame(
        columns=['Test Name', 'Test Count', 'People Count', 'True Positive', 'False Positive',
                 'True Negative', 'False Negative', 'True Positive Rate',
                 'True Negative Rate', 'Positive Predictive Value',
                 'Negative Predictive Value', 'Regret', 'Revenue', 'Actual Average EV Lift',
                 'Measured Average EV Lift', 'Actual Total EV Lift/Million',
                 'Measured Total EV Lift/Million'])

    axes_count = 0
    m_procs = []
    for rule in decision_rules:
        if plot:
            axis_num = axes_count
        else:
            axis_num = None
        # Actually Run the tests
        m_procs.append(Process(target=run_test, args=(rule, q, plot_q, mrr, n, p_baseline, max_tests, axis_num, seed)))

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
            m_axis = get_axis(label_count)
            m_axis.set_title(rule.__name__)
            m_axis.set_xlabel('True Difference in EV')
            m_axis.set_ylabel('Measured Difference in EV')
            label_count += 1

        # Show plots in a maximized window

        fig_manager = plt.get_current_fig_manager()
        #fig_manager.window.showMaximized()

        finished_count = 0
        while finished_count < len(m_procs):

            val = plot_q.get()
            if val[0] == 'finished':
                finished_count += 1
            else:
                x, y, axis, color = val

                get_axis(axis).scatter(x, y, color=color)
                get_axis(axis).figure.canvas.draw()
                plt.pause(.05)
        filename = 'results/comparison/' + time.strftime('%Y%m%d_%H-%M_') + 'Agg Results.png'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        multi_plt.savefig(filename)

    for p in m_procs:
        test_data, test_name = q.get()
        test_result = pd.DataFrame(test_data, columns=list(ind_test_results.columns))
        ind_test_results = pd.concat([ind_test_results, test_result], ignore_index=True)

        true_positive = test_result[(test_result['Choice'] == 'B') & (test_result['Actual Winner'] == 'B')]
        false_positive = test_result[(test_result['Choice'] == 'B') & (test_result['Actual Winner'] == 'A')]
        true_negative = test_result[(test_result['Choice'] == 'A') & (test_result['Actual Winner'] == 'A')]
        false_negative = test_result[(test_result['Choice'] == 'A') & (test_result['Actual Winner'] == 'B')]
        temp_agg = [[test_name, len(test_result), test_result['Total Number'].sum(), len(true_positive),
                     len(false_positive),
                     len(true_negative), len(false_negative),
                     (len(true_positive) / (len(true_positive) + len(false_negative)) if (
                         len(true_positive) + len(false_negative) != 0) else 0),
                     (len(true_negative) / (len(true_negative) + len(false_positive)) if (
                         len(true_negative) + len(false_positive) != 0) else 0),
                     (len(true_positive) / (len(true_positive) + len(false_positive)) if (len(true_positive) + len(
                         false_positive)) != 0 else 0),
                     (len(true_negative) / (len(true_negative) + len(false_negative)) if (len(true_negative) + len(
                         false_negative)) != 0 else 0),
                     test_result.Regret.sum(), test_result[['A Revenue', 'B Revenue']].sum().sum(),
                     (test_result[test_result['Choice'] == 'B']['EV B'] / test_result[test_result['Choice'] == 'B'][
                         'EV A'] - 1).mean(),
                     (test_result[test_result['Choice'] == 'B']['EV B Measured'] / test_result[test_result[
                                                                                                   'Choice'] == 'B'][
                         'EV A Measured'] - 1).mean(),
                     (test_result[test_result['Choice'] == 'B']['EV B'] - test_result[test_result['Choice'] == 'B'][
                         'EV A']).sum()/test_result['Total Number'].sum()*1000000,
                     (test_result[test_result['Choice'] == 'B']['EV B Measured'] - test_result[test_result[
                                                                                                   'Choice'] == 'B'][
                         'EV A Measured']).sum()/test_result['Total Number'].sum()*1000000
                     ]]
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
