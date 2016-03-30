import os
import numpy as np
from cec2013_mmop import functions as f

__dir__ = os.path.dirname(os.path.realpath(__file__))


class _CompositeFunction():
    """
    Class represents a composite function
    """

    def __init__(self, funcs, lmbda, sigma, dimension):
        self._dimension = dimension
        self._funcs = funcs
        self._nofuncs = len(funcs)
        self._O = []
        self._lmbda = lmbda
        self._bias = np.asarray([0] * self._nofuncs)
        self._sigma = sigma
        self._M = [[] for _ in range(self._nofuncs)]
        self._fmaxi = np.array([0.0] * self._nofuncs)

    def load_optimas(self, prefix):
        """
        Load optimas
        """
        filename = '{0}/../data/{1}_M_D{2}_opt.dat'\
                   .format(__dir__, prefix, self._dimension)

        with open(filename, "r") as ins:
            for idx, line in enumerate(ins):
                if idx >= self._nofuncs:
                    break
                self._O.append([float(
                    x) for x in filter(lambda x: x != '\n', line.split('\t'))])

        self._O = np.asarray(self._O)

    def load_rotation_matrix(self, prefix):
        """
        Load rotation matrix
        """
        filename = '{0}/../data/{1}_M_D{2}.dat'\
                   .format(__dir__, prefix, self._dimension)
        with open(filename, "r") as ins:
            for idx, line in enumerate(ins):
                i = int(idx / self._dimension)
                if i >= self._nofuncs:
                    break
                arr = [float(x)
                       for x in filter(lambda x: x != '\n', line.split('\t'))]
                self._M[i].append(arr)

        self._M = np.asarray(self._M)

    def load_rotation_matrix_identity(self):
        self._M = np.array([np.identity(self._dimension)] * self._nofuncs)

    def calculate_fmaxi(self):
        x5 = np.array([5.0] * self._dimension)
        for idx, func in enumerate(self._funcs):
            self._fmaxi[idx] = func(self.transform_z_noshift(x5, idx))

    def calculate_weights(self, x):
        s = self._sigma**2
        sum_w = np.sum((x - self._O)**2, axis=1)
        w = np.e**(-1.0 * sum_w / (2.0 * self._dimension * s))

        maxidx = w.argmax()
        maxi = w[maxidx]

        sw = np.asarray([1. - maxi**10] * len(w))
        sw[maxidx] = 1.0

        w = w * sw
        sum_w = np.sum(w)
        if sum_w == 0.:
            w = [1. / self._nofuncs] * len(w)
        else:
            w = w / sum_w

        return w

    def transform_z(self, x, idx):
        tmpx = (x - self._O[idx]) / self._lmbda[idx]
        return (np.matrix(tmpx) * np.matrix(self._M[idx])).A1

    def transform_z_noshift(self, x, idx):
        tmpx = x / self._lmbda[idx]

        return (np.matrix(tmpx) * np.matrix(self._M[idx])).A1

    def evaluate(self, x):
        if len(x) != self._dimension:
            raise ValueError('Invalid dimension')

        w = self.calculate_weights(x)
        fi = []
        c = 2000.
        for idx, func in enumerate(self._funcs):
            fi.append(func(self.transform_z(x, idx)))

        fi = np.asarray(fi)
        result = np.sum(w * (c * fi / self._fmaxi + self._bias))
        return -1.0 * result


_cf1_data = {
    'name': 'CF1',
    'funcs':
    [f.griewank, f.griewank, f.weierstrass, f.weierstrass, f.sphere, f.sphere],
    'lambda': np.array([1., 1., 8., 8., 1. / 5., 1. / 5.]),
    'sigma': np.array([1.] * 6)
}

_cf2_data = {
    'name': 'CF2',
    'funcs': [f.rastrigin, f.rastrigin, f.weierstrass, f.weierstrass,
              f.griewank, f.griewank, f.sphere, f.sphere],
    'lambda':
    np.array([1., 1., 10., 10., 1. / 10., 1. / 10., 1. / 7., 1. / 7.]),
    'sigma': np.array([1.] * 8)
}

_cf3_data = {
    'name': 'CF3',
    'funcs': [f.ef8f2, f.ef8f2, f.weierstrass, f.weierstrass, f.griewank,
              f.griewank],
    'lambda': np.array([1. / 4., 1. / 10., 2., 1., 2., 5.]),
    'sigma': np.array([1., 1., 2., 2., 2., 2.])
}

_cf4_data = {
    'name': 'CF4',
    'funcs': [f.rastrigin, f.rastrigin, f.ef8f2, f.ef8f2, f.weierstrass,
              f.weierstrass, f.griewank, f.griewank],
    'lambda':
    np.array([4., 1., 4., 1., 1. / 10., 1. / 5., 1. / 10., 1. / 40.]),
    'sigma': np.array([1., 1., 1., 1., 1., 2., 2., 2.])
}


def _create_cf_obj(cf_data, d):
    if 'cache' not in cf_data:
        cf_data['cache'] = {}

    cf_obj = None

    if d in cf_data['cache']:
        cf_obj = cf_data['cache'][d]

    else:
        cf_obj = _CompositeFunction(cf_data['funcs'], cf_data['lambda'],
                                    cf_data['sigma'], d)
        cf_obj.load_optimas(cf_data['name'])

        if cf_data['name'] in ['CF1', 'CF2']:
            cf_obj.load_rotation_matrix_identity()
        else:
            cf_obj.load_rotation_matrix(cf_data['name'])

        cf_obj.calculate_fmaxi()
        cf_data['cache'][d] = cf_obj

    return cf_obj


def cf1(x):
    return _create_cf_obj(_cf1_data, len(x)).evaluate(x)


def cf2(x):
    return _create_cf_obj(_cf2_data, len(x)).evaluate(x)


def cf3(x):
    return _create_cf_obj(_cf3_data, len(x)).evaluate(x)


def cf4(x):
    return _create_cf_obj(_cf4_data, len(x)).evaluate(x)
