# cython: c_string_type=str, c_string_encoding=ascii
# vim: syntax=pyrex

# Copyright (C) 2018-2019 Bho Matthiesen, Karl-Ludwig Besser
# 
# This program is used in the article:
# 
# Bho Matthiesen, Alessio Zappone, Karl-L. Besser, Eduard A. Jorswieck, and
# Merouane Debbah, "A Globally Optimal Energy-Efficient Power Control Framework
# and its Efficient Implementation in Wireless Interference Networks,"
# submitted to IEEE Transactions on Signal Processing
# 
# License:
# This program is licensed under the GPLv2 license. If you in any way use this
# code for research that results in publications, please cite our original
# article listed above.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.


from libcpp.vector cimport vector
from libcpp.string cimport string
from cython import boundscheck, wraparound
cimport libcpp
import numpy as np
cimport numpy as cnp
import h5py

cdef extern from "wsee_lambert.h":
    # https://stackoverflow.com/a/38567389/8372023
    #cdef enum Status "WSEE<DIM>::Status":
    #    Optimal, Unsolved, Infeasible

    # https://stackoverflow.com/a/41200186/8372023 
    cdef cppclass vtype "std::array<DIM, double>":
        double operator[](size_t)
        double* data()
        size_t size()

    cdef cppclass wsee`'`'DIM "WSEE<DIM>":
        double mu[DIM]
        double psi[DIM]
        double alpha[DIM]
        double beta[DIM][DIM]
        double sigma[DIM]

        libcpp.bool useRelTol
        libcpp.bool disableReduction

        vtype xopt
        double optval
        unsigned long long iter
        unsigned long long lastUpdate
        const char* statusStr
        double runtime

        wsee`'`'DIM`'`'()
        void setUB(double)
        void setLB(double)
        void setPrecision(double)
        void optimize(libcpp.bool)
        double getEpsilon()


cdef class WSEE`'`'DIM:
    cdef wsee`'`'DIM *_thisptr
    cdef cnp.dtype result_dt

    @wraparound(False)
    @boundscheck(False)
    def __cinit__(self, double mu, double psi):
        self._thisptr = new wsee`'`'DIM`'`'()

        self._thisptr.setLB(0)
        self._thisptr.setPrecision(1e-2)
        self._thisptr.useRelTol = True
        self._thisptr.disableReduction = True

        for i in range(DIM):
            self._thisptr.mu[i] = mu
            self._thisptr.psi[i] = psi

        self.result_dt = np.dtype([
            ("Objective Value", np.float32),
            ("xopt", np.float32, DIM),
            ("Status", h5py.special_dtype(vlen=str)),
            ('Iterations', np.uint64),
            ('Last Update', np.uint64),
            ('Epsilon', np.float32),
            ('Relative Tolerance', np.bool),
            ('Runtime', np.float32)])

    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr

    @wraparound(False)
    @boundscheck(False)
    def setChan(self, double[:] alpha, double[:,::1] beta):
        if not (alpha.shape[0] == DIM and alpha.ndim == 1):
            raise ValueError('alpha must be 1D array with DIM elements')

        if not (beta.shape[0] == DIM and beta.shape[1] == DIM and beta.ndim == 2):
            raise ValueError("beta must be DIM()x`'DIM array")

        
        for i in range(DIM):
            self._thisptr.alpha[i] = alpha[i]

            for j in range(DIM):
                if j == i:
                    if beta[i][i] != 0.0:
                        raise ValueError('beta[i][i] must be zero')
                else:
                    self._thisptr.beta[i][j] = beta[i][j]

    @wraparound(False)
    @boundscheck(False)
    def optimizeFrom(self, double[:] xopt):
        if not (xopt.shape[0] == DIM and xopt.ndim == 1):
            raise ValueError('xopt must be 1D array with DIM elements')

        cdef double *X = self._thisptr.xopt.data()
        for i in range(DIM):
            X[i] = xopt[i]

        self._thisptr.optimize(True)


    def setPmax(self, P):
        self._thisptr.setUB(P)
    
    def setPrecision(self, epsilon):
        self._thisptr.setPrecision(epsilon)

    def optimize(self):
        self._thisptr.optimize(False)

    def result(self):
        t = self._thisptr
        return np.array([(t.optval, self.xopt(), t.statusStr, t.iter, t.lastUpdate, t.getEpsilon(), t.useRelTol, t.runtime)], dtype = self.result_dt)

    def getResultDt(self):
        return self.result_dt

    @wraparound(False)
    @boundscheck(False)
    def xopt(self):
        cdef size_t size = self._thisptr.xopt.size()
        cdef cnp.ndarray xopt = np.empty((size,))

        for i in range(size):
            xopt[i] = self._thisptr.xopt[i]

        return xopt
