import os

import numpy as np

from cec2013 import functions as f


__dir__ = os.path.dirname(os.path.realpath(__file__))


class _cfunc():
    """
    Class represents the composite functions
    """

    def __init__(self, dimension, funcs):
        self._dimension = dimension
        self._funcs = funcs
        self._nofuncs = len(funcs)

        # initialize all the variables
        self._c = 2000.0
        self._fi = np.array([0.0] * self._nofuncs)
        self._fmaxi = np.array([0.0]*self._nofuncs)
        self._f_bias = 0.0
        self._O = []
        self._M = [[] for _ in range(self._nofuncs)]

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
                self._O.append([float(x) for x in filter(lambda x: x != '\n',
                                                         line.split('\t'))])

        self._O = np.array(self._O)

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
                arr = [float(x) for x in filter(lambda x: x != '\n',
                                                line.split('\t'))]
                self._M[i].append(arr)

        self._M = np.array(self._M)

    def calculate_weights(self, x):
        s = np.array(self._sigma)**2
        sum_w = np.sum((x-self._O)**2, axis=1)
        w = np.e**(-1.0*sum_w / (2.0*self._dimension*s))

        maxidx = w.argmax()
        maxi = w[maxidx]

        sw = np.array([1.0 - maxi**10]*len(w))
        sw[maxidx] = 1.0

        w = w * sw
        sum_w = np.sum(w)
        if sum_w == 0.0:
            w = [1.0/self._nofuncs]*len(w)
        else:
            w = w/sum_w

        self._weight = np.array(w)

    def init_rotation_matrix_identity(self):
        self._M = np.array([np.identity(self._dimension)]
                           * self._nofuncs)

    def transform_z(self, x, idx):
        self._tmpx = (x - self._O[idx])/self._lambda[idx]
        self._z = (np.matrix(self._tmpx) * np.matrix(self._M[idx])).A1

    def transform_z_noshift(self, x, idx):
        self._tmpx = x/self._lambda[idx]
        self._z = (np.matrix(self._tmpx) * np.matrix(self._M[idx])).A1

    def calculate_fmaxi(self):
        x5 = np.array([5.0]*self._dimension)
        for idx, func in enumerate(self._funcs):
            self.transform_z_noshift(x5, idx)
            self._fmaxi[idx] = func(self._z)

    def evaluate(self, x):
        x = np.array(x)
        if len(x) != self._dimension:
            raise ValueError('Invalid dimension')

        self.calculate_weights(x)
        for idx, func in enumerate(self._funcs):
            self.transform_z(x, idx)
            self._fi[idx] = func(self._z)
        result = np.sum(self._weight * (self._c * self._fi /
                        self._fmaxi + self._bias))
        return -1.0*result + self._f_bias


class cf1(_cfunc):
    """
    First composite function
    """

    def __init__(self, dimension):
        funcs = [f.griewank, f.griewank, f.weierstrass, f.weierstrass,
                 f.sphere, f.sphere]
        _cfunc.__init__(self, dimension, funcs)

        _cfunc._lambda = np.array([1.0, 1.0, 8.0, 8.0, 1.0/5.0, 1.0/5.0])
        _cfunc._sigma = np.array([1.0]*self._nofuncs)
        _cfunc._bias = np.array([0.0]*self._nofuncs)
        _cfunc._weight = np.array([0.0]*self._nofuncs)
        _cfunc.load_optimas(self, 'CF1')
        _cfunc.init_rotation_matrix_identity(self)
        _cfunc.calculate_fmaxi(self)


class cf2(_cfunc):
    """
    Second composite function
    """

    def __init__(self, dimension):
        funcs = [f.rastrigin, f.rastrigin, f.weierstrass, f.weierstrass,
                 f.griewank, f.griewank, f.sphere, f.sphere]
        _cfunc.__init__(self, dimension, funcs)

        _cfunc._lambda = np.array([1.0, 1.0, 10.0, 10.0, 1.0/10.0, 1.0/10.0,
                                   1.0/7.0, 1.0/7.0])
        _cfunc._sigma = np.array([1.0]*self._nofuncs)
        _cfunc._bias = np.array([0.0]*self._nofuncs)
        _cfunc._weight = np.array([0.0]*self._nofuncs)
        _cfunc.load_optimas(self, 'CF2')
        _cfunc.init_rotation_matrix_identity(self)
        _cfunc.calculate_fmaxi(self)


class cf3(_cfunc):
    """
    Third composite function
    """

    def __init__(self, dimension):
        funcs = [f.ef8f2, f.ef8f2, f.weierstrass, f.weierstrass,
                 f.griewank, f.griewank]
        _cfunc.__init__(self, dimension, funcs)

        _cfunc._lambda = np.array([1.0/4.0, 1.0/10.0, 2.0, 1.0, 2.0, 5.0])
        _cfunc._sigma = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
        _cfunc._bias = np.array([0.0]*self._nofuncs)
        _cfunc._weight = np.array([0.0]*self._nofuncs)
        _cfunc.load_optimas(self, 'CF3')
        _cfunc.load_rotation_matrix(self, 'CF3')
        _cfunc.calculate_fmaxi(self)


class cf4(_cfunc):
    """
    Fourth composite function
    """

    def __init__(self, dimension):
        funcs = [f.rastrigin, f.rastrigin, f.ef8f2, f.ef8f2,
                 f.weierstrass, f.weierstrass, f.griewank, f.griewank]
        _cfunc.__init__(self, dimension, funcs)

        _cfunc._lambda = np.array([4.0, 1.0, 4.0, 1.0, 1.0/10.0, 1.0/5.0,
                                   1.0/10.0, 1.0/40.0])
        _cfunc._sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        _cfunc._bias = np.array([0.0]*self._nofuncs)
        _cfunc._weight = np.array([0.0]*self._nofuncs)
        _cfunc.load_optimas(self, 'CF4')
        _cfunc.load_rotation_matrix(self, 'CF4')
        _cfunc.calculate_fmaxi(self)
