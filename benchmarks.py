import numpy as np
from cec2013 import functions
from cec2013 import cfunctions

class _benchmark():
    """
    Represents the benchmark
    """
    def __init__(self, **kwargs):
        self._func = kwargs.pop('f')
        self._fopt = kwargs.pop('opt')
        self._foptno = kwargs.pop('nopt')
        self._rho = kwargs.pop('rho')
        self._maxcalls = kwargs.pop('mc')
        self._upper_bound = kwargs.get('b')[0]
        self._lower_bound = kwargs.pop('b')[1]
        self._dimension = kwargs.pop('d')

    def _dist(a, b):
        return np.sqrt(np.sum(np.array(a) - np.array(b))**2)

    def _get_seeds(spop, radius):
        seeds = []
        for ind in spop:
            found = any(_benchmark._dist(ind, x) <= radius for x in seeds)
            if not found:
                seeds.append(ind)
                yield ind

    def evaluate(self, x):
        x = np.array(x)
        return self._func(x)

    def optima_count(self, pop, accuracy):
        """
        Get number of global optima in the population
        """
        spop = sorted(pop, key=lambda k: self.evaluate(k), reverse=True)
        count = 0
        for ind in _benchmark._get_seeds(spop, self._rho):
            seed_fit = self.evaluate(ind)
            if np.abs(seed_fit - self._fopt) <= accuracy:
                count = count + 1

            if count == self._foptno:
                break

        return count


def b1():
    return _benchmark(f=functions.himmelblau, opt=200.0, nopt=4,
                      rho=0.01, mc=50000, b=(-6.0, 6.0), d=2)


def b2():
    return _benchmark(f=functions.six_hump_camel_back, opt=1.03163, nopt=2,
                      rho=0.5, mc=50000, b=(-1.9, 1.9), d=2)


def b3():
    return _benchmark(f=functions.shubert, opt=186.731, nopt=18,
                      rho=0.5, mc=200000, b=(-10, 10), d=2)


def b4():
    return _benchmark(f=functions.vincent, opt=1.0, nopt=36,
                      rho=0.2, mc=200000, b=(-0.25, 10.0), d=2)


def b5():
    return _benchmark(f=functions.shubert, opt=2709.0935, nopt=81,
                      rho=0.5, mc=200000, b=(-10.0, 10.0), d=3)
