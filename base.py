#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base classes for multi-objective multi-fidelity (MOMF) problem suite from:

"A Test Suite for Multi-objective Multi-fidelity Optimization"

Authors:    Angus Kenny, UNSW Canberra,
            Tapabata Ray, UNSW Canberra,
            Hemant Kumar Singh, UNSW Canberra,
            Xiaodong Li, RMIT University.

These classes implement common methods used by the defined problems and are 
not intended to be instantiated in their own right. Please use the classes in
MOMF_problems.py (or, RE_problems.py and DTLZ_problems.py for the original
problems the MOMF problems are based on).

These problem classes were developed to interface with the pymoo Problem 
class. Therefore the ouput of the `evaluate' method is a NxM objective 
matrix, with N sampled points and M objectives, returned as a dictionary under 
the key 'F'.

See MOMF_problems.py for a full description of use.
"""

import numpy as np
from wang_error_functions import e_r1

class _BaseProblem(object):
    """Base class for all problems.
    """
    def __init__(self, *args, **kwargs):
        self.n_var = len(self._vars)
        super().__init__(*args, **kwargs)
    def get_bound_arrays(self):
        """Converts defined variable values into pymoo compatible
        bound arrays.

        Returns:
            np.array, np.array: array of variable minima and maxima.
        """
        x_min = np.array([v[0] for k,v in self._vars.items()])
        x_max = np.array([v[-1] for k,v in self._vars.items()])
        return x_min, x_max

    def evaluate(self, x):
        """Performs a problem evaluation.

        Args:
            x (np.array): NxM variable matrix with N sampling points 
                          and M variables.

        Returns:
            dict: NxO objective matrix with N sampled points 
                  and O objectives.
        """
        F = self._evaluate_obj(x)
        return {'F': F}      
        
        
class _MF_Base(object):
    """Base class for multi-fidelity problem transformation. Should only be
    used in conjunction with a problem class when creating instances.
    
    Default error function is `e_r1`, but can be changed when instantiating
    or using the `set_error_fn` method.
    """
    def __init__(self, *args, err_fn=e_r1, **kwargs):
        self.err_fn = err_fn.__func__
        super().__init__(*args, **kwargs)
    
    def set_error_fn(self, err_fn):
        """Sets the error function.

        Args:
            err_fn (staticmethod): error function defined 
            in wang_error_functions.py.
        """
        self.err_fn = err_fn.__func__
        
    def evaluate_MF(self, x, **kwargs):
        """Performs a multi-fidelity problem evaluation. Applies the specified
        error function to the output of `evaluate(x)` and returns modified
        objective matrix.

        Args:
            x (np.array): NxM variable matrix with N sampling points 
                          and M variables.

        Returns:
            dict: NxO (modified) objective matrix with N sampled points 
                  and O objectives.
        """
        n = x.shape[0]
        result = self.evaluate(x)
        
        # scale x between -1 and 1 for error functions
        x_min, x_max = self.get_bound_arrays()
        x_s = (1 - -1) * (x - x_min)/(x_max - x_min) + -1
        
        # apply error function (if applicable) to evaluated values
        for f in range(self.n_obj):
            f_kw = f'f{f+1}'
            Phi_kw = f'Phi_{f_kw}'
            if Phi_kw not in kwargs:
                result['F'][:,f] = np.NaN * np.ones(n)
            elif kwargs[Phi_kw] > 0:                
                err = (self.mf_def[self.err_fn.__name__]['scale'][f_kw]
                       * self.err_fn(x_s, self.mf_def[self.err_fn.__name__]
                                     ['Phi'][kwargs[Phi_kw]]))
                
                result['F'][:,f] = result['F'][:,f] + err
            
        return result