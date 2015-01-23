import theano.tensor as T


def log_sig(x):
    return 1 / (1 + T.exp(-x))


def tan_sig(x):
    return 2 / (1 + T.exp(-2 * x)) - 1


def pure_lin(x):
    return x


def sat_lin(x):
    return T.clip(x, 0, 1)


def sat_lins(x):
    return T.clip(x, -1, 1)


def pos_lin(x):
    return T.switch(T.lt(x, 0), 0, x)


