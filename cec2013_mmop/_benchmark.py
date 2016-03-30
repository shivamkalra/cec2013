import numpy as np
import functions as f
import cfunctions as cf
import operator
from functools import reduce


class Benchmark():
    """
    Represents the benchmark
    """

    def __init__(self, **kwargs):
        self._func = kwargs.pop('f')
        self._fopt = kwargs.pop('opt')
        self._foptno = kwargs.pop('nopt')
        self._rho = kwargs.pop('rho')
        self.max_calls = kwargs.pop('mc')
        self.call_count = 0
        self.lower_bound = kwargs.pop('lb')
        self.upper_bound = kwargs.get('ub')
        self.dimension = kwargs.pop('d')

    def _dist(a, b):
        return np.sqrt(np.sum(np.array(a) - np.array(b))**2)

    def _get_seeds(spop, radius):
        seeds = []
        for i, ind in enumerate(spop):
            found = any(Benchmark._dist(ind, x) <= radius for x in seeds)
            if not found:
                seeds.append(ind)
                yield i

    def evaluate(self, x):
        self.call_count = self.call_count + 1
        return self._func(np.asarray(x))

    def optima_count(self, pop, accuracy, values=None):
        """
        Get number of global optima in the population
        """
        if values is None:
            values = [self.evaluate(k) for k in pop]
        sortedData = sorted(zip(values, pop))

        spop = [x for _, x in sortedData]
        svals = [y for y, _ in sortedData]

        count = 0
        for ind in Benchmark._get_seeds(spop, self._rho):
            seed_fit = svals[ind]
            if np.abs(seed_fit - self._fopt) <= accuracy:
                count = count + 1

            if count == self._foptno:
                break

        return count


_funcs = [f.five_uneven_peak_trap, f.equal_maxima, f.uneven_decreasing_maxima,
          f.himmelblau, f.six_hump_camel_back, f.shubert, f.vincent, f.shubert,
          f.vincent, f.modified_rastrigin_all, cf.cf1, cf.cf2, cf.cf3, cf.cf3,
          cf.cf4, cf.cf3, cf.cf4, cf.cf3, cf.cf4, cf.cf4]

_dims = [1, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 5, 5, 10, 10, 20]

_max_fes = [[50000] * 5, [200000], [200000], [400000], [400000], [200000] * 4,
            [400000] * 7]
_max_fes = reduce(operator.add, _max_fes)

_fgoptima = [[200.], [1.], [1.], [200.], [1.03163], [186.731], [1.],
             [2709.0935], [1.], [-2.], [0] * 10]
_fgoptima = reduce(operator.add, _fgoptima)

_nopt = [2, 5, 1, 4, 2, 18, 36, 81, 216, 12, 6, 8, 6, 6, 8, 6, 8, 6, 8, 8]

_rho = [[0.01] * 4, [0.5] * 2, [0.2], [0.5], [0.2], [0.01] * 11]
_rho = reduce(operator.add, _rho)


def _get_lb(fno):
    lb = []
    dim = _dims[fno - 1]
    if fno in [1, 2, 3, 10]:
        lb = [0] * dim
    elif fno == 4:
        lb = [-6] * dim
    elif fno == 5:
        lb = [-1.9, -1.1]
    elif fno in [6, 8]:
        lb = [-10] * dim
    elif fno in [7, 9]:
        lb = [0.25] * dim
    elif fno >= 11:
        lb = [-5] * _dims[fno - 1]
    return np.asarray(lb)


def _get_ub(fno):
    ub = []
    dim = _dims[fno - 1]

    if fno == 1:
        ub = [30]
    elif fno in [2, 3, 10]:
        ub = [1] * dim
    elif fno == 4:
        ub = [6] * dim
    elif fno == 5:
        ub = [1.9, 1.1]
    elif fno in [6, 7, 8, 9]:
        ub = [10] * dim
    elif fno >= 11:
        ub = [5] * dim
    return np.asarray(ub)


_cec2013Benchmark_suite = []
for i in range(len(_funcs)):
    b = Benchmark(f=_funcs[i],
                  opt=_fgoptima[i],
                  nopt=_nopt[i],
                  rho=_rho[i],
                  mc=_max_fes[i],
                  lb=_get_lb(i + 1),
                  ub=_get_ub(i + 1),
                  d=_dims[i])
    _cec2013Benchmark_suite.append(b)
