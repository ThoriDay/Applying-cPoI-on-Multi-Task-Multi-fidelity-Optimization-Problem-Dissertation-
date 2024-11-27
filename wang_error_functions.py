#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Python implementation of the set of generic functions used to model the 
various ways that noise is created by reducing the fidelity of a numerical
simulation. For use with the multi-objective multi fidelity (MOMF)
problem suite from:

"A Test Suite for Multi-objective Multi-fidelity Optimization"

Authors:    Angus Kenny, UNSW Canberra,
            Tapabata Ray, UNSW Canberra,
            Hemant Kumar Singh, UNSW Canberra,
            Xiaodong Li, RMIT University.

These functions are based on the work in the paper "A Generic Test Suite for
Evolutionary Multifidelity Optimisation" and from code found in 

https://github.com/HandingWang/MFB

Authors:    Handing Wang, Member IEEE,
            Youchu Jin, Fellow IEEE,
            John Doherty

(THE FOLLOWING DOC TEXT IS COPIED FROM MATLAB CODE, UPDATED FOR PYTHON)

A test function f(x) can be modified to produce its "noisy" equivalent
f'(x,p) using the formula:

    f'(x,p) = f(x) + e(x,p)

Where x is a solution, p is a value in {0,1,...,10000} representing the 
fidelity required (10000 is full fidelity) and e(x,p) is the error function
which generates the noise.

There are 13 different error functions, grouped into three categories,
representing the different types of errors that can be obtained,
    - Type I    (e_1 - e_7):    resolution errors
    - Type II   (e_8 - e_11):   stochastic errors
    - Type III  (e_12 - e_13):  instability errors

Each of these error functions are implemented in this file and can be
accessed by setting the e_type parameter.

(NOTE: changed sigma to equal nu instead of 0.1nu for stochastic)

@param xx       : input value matrix with rows being candidates to be
                    evaluated and columns being the input values
@param phi      : value in {0,1,...,10000} indicating fidelity, with 
                    10000 being maximum fidelity
@param e_type   : type of error function required in {1,...,10}

@return e       : column vector of noise values to be added to f(x)
"""

import numpy as np

# e_r^1 for MFB1, MFB4, MFB6
@staticmethod
def e_r1(x, phi):
    n = x.shape[0]
    d = x.shape[1]
    theta = 1 - 0.0001 * phi
    a = theta * np.ones((n, d))
    w = 10 * np.pi * theta * np.ones((n, d))
    b = 0.5 * np.pi * theta * np.ones((n, d))
    e = np.sum(a * np.cos(w * x + b + np.pi), axis=1)
    return e

# e_r^2 for MFB2, MFB5
@staticmethod
def e_r2(x, phi):
    n = x.shape[0]
    d = x.shape[1] 
    theta = np.exp(-0.00025*phi)
    a = theta * np.ones((n,d))
    w = 10 * np.pi * theta * np.ones((n,d))
    b = 0.5 * np.pi * theta * np.ones((n,d))
    e = np.sum(a * np.cos(w * x + b + np.pi), axis=1)
    return e

# e_r^3 for MFB3
@staticmethod
def e_r3(x, phi):
    n = x.shape[0]
    d = x.shape[1] 
    if phi < 1000:
        theta = 1 - 0.0002 * phi
    elif phi < 2000:
        theta = 0.8
    elif phi < 3000:
        theta = 1.2 - 0.0002 * phi
    elif phi < 4000:
        theta = 0.6
    elif phi < 5000:
        theta = 1.4 - 0.0002 * phi
    elif phi < 6000:
        theta = 0.4
    elif phi < 7000:
        theta = 1.6 - 0.0002 * phi
    elif phi < 8000:
        theta = 0.2
    elif phi < 9000:
        theta = 1.8 - 0.0002 * phi
    else:
        theta = 0
    a = theta * np.ones((n,d))
    w = 10 * np.pi * theta * np.ones((n,d))
    b = 0.5 * np.pi * theta * np.ones((n,d))
    e = np.sum(a * np.cos(w * x + b + np.pi), axis=1)
    return e

# e_r^4 for MFB7 (requires x_b)
@staticmethod
def e_r4(x, phi, x_b):
    n = x.shape[0]
    d = x.shape[1] 
    theta = 1 - 0.0001 * phi
    psi = 1 - np.abs(x - x_b)
    a = theta * np.ones((n,d)) * psi
    w = 10 * np.pi * theta * np.ones((n,d))
    b = 0.5 * np.pi * theta * np.ones((n,d))
    e = np.sum(a * np.cos(w * x + b + np.pi), axis=1)
    return e

# e_s^1 for MFB8
@staticmethod
def e_s1(x, phi):
    n = x.shape[0]
    nu = 1 - 0.0001 * phi
    sigma = 0.1 * nu
    mu = 0
    e = np.random.rand(n) * sigma + mu
    return e

# e_s^2 for MFB9
@staticmethod
def e_s2(x, phi):
    n = x.shape[0]
    nu = np.exp(-0.0005 * phi)
    sigma = 0.1 * nu
    mu = 0
    e = np.random.rand(n) * sigma + mu
    return e

# e_s^3 for MFB10
@staticmethod
def e_s3(x, phi):
    n = x.shape[0]
    d = x.shape[1] 
    nu = 1 - 0.0001 * phi
    gamma = np.sum(1 - np.abs(x), axis=1)
    sigma = 0.1 * nu
    mu = (sigma / d) * gamma
    e = np.random.rand(n) * sigma + mu
    return e

# e_s^4 for MFB11
@staticmethod
def e_s4(x, phi):
    n = x.shape[0]
    d = x.shape[1] 
    nu = np.exp(-0.0005 * phi)
    gamma = np.sum(1 - np.abs(x), axis=1)
    sigma = 0.1 * nu
    mu = (sigma / d) * gamma
    e = np.random.rand(n) * sigma + mu
    return e
    
# e_ins^1 for MFB12
@staticmethod
def e_ins1(x, phi):
    n = x.shape[0]
    d = x.shape[1] 
    r = np.random.rand(n)
    e = 10 * d * np.ones(n)
    p = 0.1 * (1 - 0.0001 * phi)
    e[np.where(r > p)] = 0
    return e

# e_ins^2 for MFB13
@staticmethod
def e_ins2(x, phi):
    n = x.shape[0]
    d = x.shape[1] 
    r = np.random.rand(n)
    e = 10 * d * np.ones(n)
    p = np.exp(-0.001 * phi - 0.1)
    e[np.where(r > p)] = 0
    return e
    