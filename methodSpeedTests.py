import numpy as np

prop_a = .5
p_a = [.2,.2,.2,.4]
p_b = [.2,.2,.2,.4]


def one_draw():
    result = np.argmax(np.random.multinomial(1, np.concatenate([prop_a*np.array(p_a), (1-prop_a)*np.array(p_b)])))
    if result <4:
        return result
    return result - 4


def multi_draw():
    my_sample = np.argmax(np.random.binomial(1, [prop_a, 0]))
    if my_sample == 0:
        return np.argmax(np.random.multinomial(1, p_a))
    return np.argmax(np.random.multinomial(1, p_b))