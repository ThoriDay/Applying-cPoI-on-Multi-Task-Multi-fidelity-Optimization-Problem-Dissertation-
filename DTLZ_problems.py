#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python implementations of the DTLZ2 and inverted DTLZ2 multi-objective 
test functions, for use with the multi-objective multi fidelity (MOMF)
problem suite from:

"A Test Suite for Multi-objective Multi-fidelity Optimization"

Authors:    Angus Kenny, UNSW Canberra,
            Tapabata Ray, UNSW Canberra,
            Hemant Kumar Singh, UNSW Canberra,
            Xiaodong Li, RMIT University.

The problems here are based on:
"Scalable test problems for evolutionary multiobjective optimization", 
by Kalyanmoy Deb, et al. and "Performance of decomposition-based 
many-objective algorithms strongly depends on Pareto front shapes", 
by Hisao Ishibuchi, et al.

Problem definition is dependent on the input parameters and is instantiated
by the `_DTLZ_Base` class. Objective value computations for the specific 
problems are given in the `_evaluate_obj` methods in their respective class 
definitions, which inherit from the `_DTLZ_Base` class.

When instantiating `DTLZ2` and `InvertedDTLZ2` without any arguments, the 
default values of `n_obj=3`, `k=5`, `beta=1` are used to define the problem 
instances, however different instances of DTLZ2 and inverted DTLZ2 can be 
created by modifying these keyword arguments when creating a class instance.

The `_evaluate_obj` method returns a MxO matrix, where M is the number of 
sampled points and O is the number of objectives.
"""

import numpy as np
from base import _BaseProblem

class _DTLZ_Base(_BaseProblem):
    """Base class for DTLZ problems. Inherits from _BaseProblem class, but
    also initialises a DTLZ problem from given parameters and computes g and s
    values.
    """
    def __init__(self, *args, n_obj=3, k=5, beta=1, **kwargs):
        self.beta = beta
        self.n_obj = n_obj
        self.k = k
        self.n_var = n_obj + k - 1
        self._vars = {f'x{n+1}': np.array([0, 1]) 
                      for n in range(self.n_var)}
        super().__init__(*args, **kwargs)
    
    def g(self, x_m):
        return ((x_m.shape[1] 
                       + np.sum(np.power(x_m-0.5,2) 
                              - np.cos(20 * np.pi * (x_m - 0.5)), axis=1))
                ).reshape(-1,1)
    
    def s(self, x_pos):
        S = np.power(np.prod(np.cos(x_pos * np.pi / 2), axis=1), self.beta
                     ).reshape(-1,1)
        for i in range(self.n_obj-1-1, -1, -1):
            s_tmp = np.power(np.prod(np.cos(x_pos[:, :i] * np.pi / 2), axis=1) 
                     * np.sin(x_pos[:, i] * np.pi / 2), self.beta)
            S = np.hstack((S, s_tmp.reshape(-1,1)))
            
        return S
          
            
class DTLZ2(_DTLZ_Base):
    '''
    One variable per objective except for last which is a function of 
    k sub-variables. These k variables are passed to g(x)
    
    _vars is defined in call to __init__
    '''
    def _evaluate_obj(self, x):
        
        x_pos, x_m = x[:, :self.n_obj-1], x[:, self.n_obj-1:]
        
        G_xm = self.g(x_m)
        
        S_xpos = self.s(x_pos)
        
        return (1 + G_xm) * S_xpos


class InvertedDTLZ2(_DTLZ_Base):
    '''
    One variable per objective except for last which is a function of 
    k sub-variables. These k variables are passed to g(x)
    
    _vars is defined in call to __init__
    '''
    def _evaluate_obj(self, x):
        
        x_pos, x_m = x[:, :self.n_obj-1], x[:, self.n_obj-1:]
        
        G_xm = self.g(x_m)
        
        S_xpos = self.s(x_pos)
        
        return (1 + G_xm) * (1 - S_xpos)