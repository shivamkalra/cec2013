import numpy as np


def five_uneven_peak_trap(x):
    """
    Variable ranges in: x in [0, 30]
    No. of global peaks: 2
    No. of local peaks: 3
    """

    result = -1
    if x >= 0 and x < 2.5:
        result = 80 * (2.5 - x)

    elif x >= 2.5 and x < 5.0:
        result = 64 * (x - 2.5)

    elif x >= 5.0 and x < 7.5:
        result = 64 * (7.5 - x)

    elif x >= 7.5 and x < 12.5:
        result = 28 * (x - 7.5)

    elif x >= 12.5 and x < 17.5:
        result = 28 * (17.5 - x)

    elif x >= 17.5 and x < 22.5:
        result = 32 * (x - 17.5)

    elif x >= 22.5 and x < 27.5:
        result = 32 * (27.5 - x)

    elif x >= 27.5 and x <= 30:
        result = 80 * (x - 27.5)

    return result


def equal_maxima(x):
    return np.sin(5 * np.pi * x)**6


def uneven_decreasing_maxima(x):
    return np.exp(-2 * np.log(2) * ((x - 0.08) / 0.854)**2) \
        * np.sin(5 * np.pi * (x**0.75 - 0.05))**6


def himmelblau(x):
    return 200 - (x[0]**2 + x[1] - 11)**2 \
        - (x[0] + x[1]**2 - 7)**2


def six_hump_camel_back(x):
    return -((4 - 2.1 * x[0]**2 + (x[0]**4) / 3) * x[0]**2 + x[0] * x[1] +
             (4 * x[1]**2 - 4) * x[1]**2)


def shubert(x):
    j = np.asarray(range(1, 6))
    return -np.multiply.reduce([sum(j * np.cos((j + 1) * xx + j)) for xx in x])


def vincent(x):
    return sum(np.sin(10 * np.log(x))) / len(x)


def modified_rastrigin_all(x):
    MMP = 0
    dmap = {2: [3, 4],
            8: [1, 2, 1, 2, 1, 3, 1, 4],
            16: [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 4]}
    d = len(x)
    if d in dmap:
        MMP = dmap[d]

    MMP = np.asarray(MMP)
    return -sum(10 + 9 * np.cos(2 * np.pi * MMP * x))


def sphere(x):
    # Please notice there is no use to rotate a sphere function, with rotation
    # here just for a similar structure as other functions and easy
    # programming
    return np.sum(x**2)


def griewank(x):
    d = len(x)
    m = np.multiply.reduce(np.cos(x / np.sqrt(range(1, d + 1))))
    s = sum(x**2 / 4000.0)
    return 1.0 + s - m


def rastrigin(x):
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10.0)


def _w(v, c1, c2):
    return sum(c1 * np.cos(c2 * v))


def weierstrass(x):
    x = x + 0.5
    a, b, kmax = 0.5, 3.0, 20

    seq = np.array(range(kmax + 1))
    c1 = a**seq
    c2 = 2.0 * np.pi * (b**seq)
    return sum([_w(e, c1, c2) for e in x]) - _w(0.5, c1, c2) * len(x)


def ef8f2(xx):
    l = len(xx)
    x = 1 + xx
    y = 1 + np.insert(xx[1:l], l - 1, xx[0])

    f2 = 100.0 * (x * x - y)**2 + (1.0 - x)**2
    f = 1.0 + ((f2**2) / 4000.0 - np.cos(f2))
    return sum(f)
