#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python implementations of the RE multi-objective test function set, for 
use with the multi-objective multi fidelity (MOMF) problem suite from:

"A Test Suite for Multi-objective Multi-fidelity Optimization"

Authors:    Angus Kenny, UNSW Canberra,
            Tapabata Ray, UNSW Canberra,
            Hemant Kumar Singh, UNSW Canberra,
            Xiaodong Li, RMIT University.

The problems here are based on a subset of problems from:
"An easy-to-use real-world multi-objective optimization problem suite", 
by Ryoji Tanabe and Hisao Ishibuchi.

All problems have continuously valued variables and are bound-constrained 
only. The definitions have two attributes specifying number of objectives 
(`n_obj`) and variable bounds (`_vars`), defined as a dictionary with 
key/value pairs consisting of the variable name and a numpy array with two
values, specifying the lower- and upper-bound respectively. Any other
class attributes are just constants from the problem definition itself.

Each problem class implements an `_evaluate_obj` method which takes a MxN 
matrix of sample points to evaluate, where M is the number of points and N 
is the number of variables. The method returns a MxO matrix, where M is the 
number of sampled points and O is the number of objectives.

The problem classes inherit from the `_BaseProblem` class, which implements
the `evaluate` method which inserts the output of `_evaluate_obj` into a
dictionary, and a method for extracting the lower and upper bounds as 
separate arrays.
"""

import numpy as np
from base import _BaseProblem

class RE2_4_1(_BaseProblem):
    '''
    Four bar truss design problem.
    
    f1 and f2 are the structural volume and joint displacement of the four bar
    truss, respectively. x1, x2, x3 and x4 are the length of the four bars.
    '''
    F = 10      # 10 kN
    E = 2e5     # 2e5 kN/cm^2
    L = 200     # 200 cm
    sigma = 10  # 10 kN/cm^2
    a = F/sigma
    
    n_obj = 2
    _vars = {'x1': np.array([a, 3 * a]),
             'x2': np.array([np.sqrt(2 * a), 3 * a]),
             'x3': np.array([np.sqrt(2 * a), 3 * a]),
             'x4': np.array([a, 3 * a])}
    
    def _evaluate_obj(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        
        f1 = (self.L * 
              (2 * x1 + np.sqrt(2 * x2) + np.sqrt(x3) + x4))
        f2 = ((self.F * self.L / self.E) * 
              ((2 / x1) + (2 * np.sqrt(2) / x2) 
               - (2 * np.sqrt(2) / x3) + (2 / x4)))
        
        F = np.hstack((f1.reshape(-1, 1), f2.reshape(-1, 1)))
        
        return F
        

class RE3_5_4(_BaseProblem):
    '''
    Vehicle crashworthiness design problem.
    
    f1, f2 and f3 minimise the weight, acceleration characteristics and
    toe-board instruction of the vehicle design, respectively.
    
    x1, x2, x3, x4 and x5 are all real-valued and specify the thickness of the 
    five reinforced members around the frontal structure of the vehicle.
    
    '''
    n_obj = 3
    _vars = {'x1': np.array([1, 3]),
             'x2': np.array([1, 3]),
             'x3': np.array([1, 3]),
             'x4': np.array([1, 3]),
             'x5': np.array([1, 3])}
    
    def _evaluate_obj(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]   
        x5 = x[:, 4]   
    
        f1 = (1640.2823 + 2.3573285 * x1 
              + 2.3220035 * x2 + 4.5688768 * x3 
              + 7.7213633 * x4 + 4.4559504 * x5)
        
        f2 = (6.5856 + 1.15 * x1 - 1.0427 * x2 + 0.9738 * x3 
              + 0.8364 * x4 - 0.3695 * x1 * x4 + 0.0961 * x1 * x5 
              + 0.3628 * x2 * x4 - 0.1106 * np.power(x1, 2) 
              - 0.3437 * np.power(x3, 2) + 0.1764 * np.power(x4, 2))
        
        f3 = (-0.0551 + 0.0181 * x1 + 0.1024 * x2 + 0.0421 * x3 
              - 0.0073 * x1 * x2 + 0.024 * x2 * x3 - 0.0118 * x2 * x4 
              - 0.0204 * x3 * x4 - 0.008 * x3 * x5 
              - 0.0241 * np.power(x2, 2) + 0.0109 * np.power(x4, 2))
        
        F = np.hstack((f1.reshape(-1,1), 
                       f2.reshape(-1,1), 
                       f3.reshape(-1,1)))
        
        return F
    
    
class RE3_4_7(_BaseProblem):
    '''
    Rocket injector design problem.
    
    f1, f2 and f3 minimise the maximum temperature of the injector face, the
    distance from the inlet and the maximum temperature of the post tip, 
    respectively.
    
    x1 is the hydrogen flow angle (alpha), x2 is the hydrogen area (Delta_HA),
    x3 is the oxygen area (Delta_OA) and x4 is the oxidiser post tip
    thickness (OPTT).
    '''
    n_obj = 3
    _vars = {'x1': np.array([0, 1]),
             'x2': np.array([0, 1]),
             'x3': np.array([0, 1]),
             'x4': np.array([0, 1])}
    
    def _evaluate_obj(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]  
    
        f1 = (0.692 + 0.477 * x1 - 0.687 * x2 - 0.08 * x3 - 0.065 * x4 
              - 0.167 * np.power(x1, 2) - 0.0129 * x2 * x1 
              + 0.0796 * np.power(x2, 2) - 0.0634 * x3 * x1
              - 0.0257 * x3 * x2 + 0.0877 * np.power(x3, 2)
              - 0.0521 * x4 * x1 + 0.00156 * x4 * x2 + 0.00198 * x4 * x3
              + 0.0184 * np.power(x4, 2))
    
        f2 = (0.153 + 0.322 * x1 - 0.396 * x2 - 0.424 * x3 - 0.0226 * x4 
              - 0.175 * np.power(x1, 2) - 0.0185 * x2 * x1 
              + 0.0701 * np.power(x2, 2) - 0.251 * x3 * x1
              - 0.179 * x3 * x2 + 0.0150 * np.power(x3, 2)
              - 0.0134 * x4 * x1 + 0.0296 * x4 * x2 + 0.0752 * x4 * x3
              + 0.0192 * np.power(x4, 2))
    
        f3 = (0.370 - 0.205 * x1 + 0.307 * x2 + 0.108 * x3 + 1.019 * x4
              - 0.135 * np.power(x1, 2) + 0.0141 * x2 * x1 
              + 0.0998 * np.power(x2, 2) + 0.208 * x3 * x1 - 0.0301 * x3 * x2
              - 0.226 * np.power(x3, 2) + 0.353 * x4 * x1 - 0.0497 * x4 * x3
              - 0.423 * np.power(x4, 2) + 0.202 * x2 * np.power(x1, 2) 
              - 0.281 * x3 * np.power(x1, 2) - 0.342 * np.power(x2, 2) * x1
              - 0.245 * np.power(x2, 2) * x3 + 0.281 * np.power(x3, 2) * x2
              - 0.184 * np.power(x4, 2) * x1 - 0.281 * x2 * x1 * x3)
        
        F = np.hstack((f1.reshape(-1,1), 
                       f2.reshape(-1,1), 
                       f3.reshape(-1,1)))
        
        return F