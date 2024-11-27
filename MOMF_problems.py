#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-objective multi-fidelity (MOMF) problem suite, from:

"A Test Suite for Multi-objective Multi-fidelity Optimization"

Authors:    Angus Kenny, UNSW Canberra,
            Tapabata Ray, UNSW Canberra,
            Hemant Kumar Singh, UNSW Canberra,
            Xiaodong Li, RMIT University.
         
These classes inherit from the base class `_MF_Base` and a problem definition
class (which inherits from the `_BaseProblem` class). 

Separate to the original problem definition, these classes contain the
attribute `mf_def` which is a dictionary, defining the fidelity levels and
scaling factors such that the required correlation between low- and 
high-fidelity functions is achieved. Details on how these fidelity levels
and scaling factors are computed is available in the paper.

Low-fidelity transformations are made using the error functions defined in
the file `wang_error_functions.py`. If a problem is instantiated without
any arguments, the default error function is `e_r1`, however this can be
specified using the keyword argument `err_fun` or by using the `set_err_fn` 
method. No other arguments are required for MOMF2-4-1(a-d), MOMF3-4-1 or 
MOMF3-5-1. Additional keyword arguments can be used with MOMF3-7-1 and 
MOMF3-7-2 to define specific instances of DTLZ2 and inverted DTLZ2 - more
information about this is given in `DTLZ_problems.py`. 

Once instantiated, a problem can be evaluated in high-fidelity using the
method `evaluate`, passing a MxN matrix of sample points to evaluate, where
M is the number of points and N is the number of variables.

To evaluate the problem in low-fidelity, the method `evaluate_MF` can be
used, passing the same MxN matrix and keyword arguments indicating which
fidelity level each objective should be evaluated at. For example, given a
problem instantiated as `prob`, to evaluate objective f1 at fidelity level
3 and f3 at fidelity level 0 (i.e., high-fidelity) for the samples in matrix
`x`, the following line should be used:

    prob.evaluate_MF(x, Phi_f1=3, Phi_f3=0)

or, alternatively:

    phis = {'Phi_f1': 3, 'Phi_f3': 0}
    prob.evaluate_MF(x, **phis)

Both `evaluate` and `evaluate_MF` return a dictionary containing the 
evaluated objectives in a MxO matrix, where M is the number of sampled points
and O is the number of objectives under the dictionary key 'F'. This is to
enable compatibility with the pymoo library, which was used as the main 
framework for the experiments in this paper. If the requred fidelity values 
were not specified as for all objectives as keywords for `evaluate_MF`, the 
returned matrix will contain `nan` in the columns representing the missing
objectives.
"""

from base import _MF_Base
from RE_problems import (RE2_4_1, RE3_4_7, RE3_5_4)
from DTLZ_problems import (DTLZ2, InvertedDTLZ2)

class MOMF2_4_1a(_MF_Base, RE2_4_1):
    """Two objective functions, with 4 variables. Fidelity levels such that 
    correlation is in [0.7, 1.0] for f1 and f2.
    """
    mf_def = {
        'e_r1': {'Phi'  : {1: 8436, 2: 7064, 3: 5000, },
        		 'scale': {'f1': 1.0962e+02, 'f2': 1.6447e-01, }},
        'e_r2': {'Phi'  : {1: 9178, 2: 6893, 3: 5000, },
        		 'scale': {'f1': 2.0085e+02, 'f2': 3.0882e-01, }},
        'e_r3': {'Phi'  : {1: 8266, 2: 6583, 3: 5000, },
        		 'scale': {'f1': 1.3117e+02, 'f2': 2.1674e-01, }},
        'e_s1': {'Phi'  : {1: 8697, 2: 6853, 3: 5000, },
        		 'scale': {'f1': 5.3856e+03, 'f2': 7.2515e+00, }},
        'e_s2': {'Phi'  : {1: 7715, 2: 5891, 3: 5000, },
        		 'scale': {'f1': 3.2636e+04, 'f2': 4.4158e+01, }},
        'e_s3': {'Phi'  : {1: 8697, 2: 7014, 3: 5000, },
        		 'scale': {'f1': 6.9287e+03, 'f2': 5.6141e+00, }},
        'e_s4': {'Phi'  : {1: 7605, 2: 6102, 3: 5000, },
        		 'scale': {'f1': 4.2286e+04, 'f2': 3.4426e+01, }},
        }


class MOMF2_4_1b(_MF_Base, RE2_4_1):
    """Two objective functions, with 4 variables. Fidelity levels such that 
    correlation is in [0.9, 1.0] for f1 and [0.7, 1.0] for f2.
    """
    mf_def = {
        'e_r1': {'Phi'  : {1: 8426, 2: 7054, 3: 5000, },
        		 'scale': {'f1': 3.5062e+01, 'f2': 1.6365e-01, }},
        'e_r2': {'Phi'  : {1: 9879, 2: 6913, 3: 5000, },
        		 'scale': {'f1': 6.3384e+01, 'f2': 3.1181e-01, }},
        'e_r3': {'Phi'  : {1: 8476, 2: 6993, 3: 5000, },
        		 'scale': {'f1': 4.3033e+01, 'f2': 2.1740e-01, }},
        'e_s1': {'Phi'  : {1: 8757, 2: 7224, 3: 5000, },
        		 'scale': {'f1': 1.7374e+03, 'f2': 7.4510e+00, }},
        'e_s2': {'Phi'  : {1: 7725, 2: 6122, 3: 5000, },
        		 'scale': {'f1': 1.0267e+04, 'f2': 4.4144e+01, }},
        'e_s3': {'Phi'  : {1: 8577, 2: 6983, 3: 5000, },
        		 'scale': {'f1': 1.7665e+03, 'f2': 5.5864e+00, }},
        'e_s4': {'Phi'  : {1: 7525, 2: 6022, 3: 5000, },
        		 'scale': {'f1': 1.0822e+04, 'f2': 3.4230e+01, }},
        }


class MOMF2_4_1c(_MF_Base, RE2_4_1):
    """Two objective functions, with 4 variables. Fidelity levels such that 
    correlation is in [0.7, 1.0] for f1 and [0.9, 1.0] for f2.        
    """
    mf_def = {
        'e_r1': {'Phi'  : {1: 8436, 2: 6663, 3: 5000, },
        		 'scale': {'f1': 1.0975e+02, 'f2': 3.9109e-02, }},
        'e_r2': {'Phi'  : {1: 9028, 2: 7304, 3: 5000, },
        		 'scale': {'f1': 2.0177e+02, 'f2': 7.1956e-02, }},
        'e_r3': {'Phi'  : {1: 8266, 2: 6673, 3: 5000, },
             'scale': {'f1': 1.3148e+02, 'f2': 5.0945e-02, }},
        'e_s1': {'Phi'  : {1: 8416, 2: 6773, 3: 5000, },
        		 'scale': {'f1': 5.3518e+03, 'f2': 1.8673e+00, }},
        'e_s2': {'Phi'  : {1: 7304, 2: 5881, 3: 5000, },
        		 'scale': {'f1': 3.2626e+04, 'f2': 1.1420e+01, }},
        'e_s3': {'Phi'  : {1: 8717, 2: 7134, 3: 5000, },
        		 'scale': {'f1': 6.9229e+03, 'f2': 1.5959e+00, }},
        'e_s4': {'Phi'  : {1: 7715, 2: 6112, 3: 5000, },
        		 'scale': {'f1': 4.2171e+04, 'f2': 9.7040e+00, }},
        }


class MOMF2_4_1d(_MF_Base, RE2_4_1):
    """Two objective functions, with 4 variables. Fidelity levels such that 
    correlation is approximately in [0.9, 1.0] for f1 and f2.
    """
    mf_def = {
        'e_r1': {'Phi'  : {1: 8286, 2: 6653, 3: 5000, },
        		 'scale': {'f1': 3.5214e+01, 'f2': 3.9022e-02, }},
        'e_r2': {'Phi'  : {1: 9028, 2: 6673, 3: 5000, },
        		 'scale': {'f1': 6.3633e+01, 'f2': 7.1366e-02, }},
        'e_r3': {'Phi'  : {1: 8256, 2: 6683, 3: 5000, },
        		 'scale': {'f1': 4.3210e+01, 'f2': 5.0684e-02, }},
        'e_s1': {'Phi'  : {1: 8396, 2: 6683, 3: 5000, },
        		 'scale': {'f1': 1.6908e+03, 'f2': 1.8579e+00, }},
        'e_s2': {'Phi'  : {1: 7254, 2: 5851, 3: 5000, },
        		 'scale': {'f1': 1.0274e+04, 'f2': 1.1334e+01, }},
        'e_s3': {'Phi'  : {1: 8376, 2: 6723, 3: 5000, },
        		 'scale': {'f1': 1.7751e+03, 'f2': 1.5857e+00, }},
        'e_s4': {'Phi'  : {1: 7254, 2: 5851, 3: 5000, },
        		 'scale': {'f1': 1.0810e+04, 'f2': 9.6065e+00, }},
        }

    

class MOMF3_4_1(_MF_Base, RE3_4_7):
    """Three objective functions, with 4 variables. Fidelity levels such that 
    correlation is in [0.7, 1.0] for f1, f2 and f3.
    """
    mf_def = {
        'e_r1': {'Phi'  : {1: 8286, 2: 6823, 3: 5000, },
        		 'scale': {'f1': 1.4381e-01, 'f2': 1.4975e-01, 'f3': 1.6675e-01, }},
        'e_r2': {'Phi'  : {1: 9368, 2: 6643, 3: 5000, },
        		 'scale': {'f1': 2.5430e-01, 'f2': 2.6420e-01, 'f3': 2.9649e-01, }},
        'e_r3': {'Phi'  : {1: 8386, 2: 6733, 3: 5000, },
        		 'scale': {'f1': 1.8696e-01, 'f2': 1.9949e-01, 'f3': 2.0545e-01, }},
        'e_s1': {'Phi'  : {1: 8436, 2: 6803, 3: 5000, },
        		 'scale': {'f1': 7.0386e+00, 'f2': 7.4024e+00, 'f3': 7.9370e+00, }},
        'e_s2': {'Phi'  : {1: 7324, 2: 5891, 3: 5000, },
        		 'scale': {'f1': 4.2954e+01, 'f2': 4.5103e+01, 'f3': 4.8392e+01, }},
        'e_s3': {'Phi'  : {1: 8426, 2: 6793, 3: 5000, },
        		 'scale': {'f1': 6.4054e+00, 'f2': 6.5529e+00, 'f3': 7.2184e+00, }},
        'e_s4': {'Phi'  : {1: 7314, 2: 5891, 3: 5000, },
        		 'scale': {'f1': 3.8890e+01, 'f2': 4.0045e+01, 'f3': 4.3946e+01, }},
        }

    
class MOMF3_5_1(_MF_Base, RE3_5_4):
    """Three objective functions, with 5 variables. Fidelity levels such that 
    correlation is in [0.7, 1.0] for f1, f2 and f3.
    """
    mf_def = {
        'e_r1': {'Phi'  : {1: 8436, 2: 6583, 3: 5000, },
        		 'scale': {'f1': 2.0252e+00, 'f2': 3.3006e-01, 'f3': 1.7791e-02, }},
        'e_r2': {'Phi'  : {1: 7995, 2: 6012, 3: 5000, },
        		 'scale': {'f1': 3.5352e+00, 'f2': 5.8414e-01, 'f3': 3.0836e-02, }},
        'e_r3': {'Phi'  : {1: 8246, 2: 6633, 3: 5000, },
        		 'scale': {'f1': 2.3078e+00, 'f2': 3.8555e-01, 'f3': 2.0860e-02, }},
        'e_s1': {'Phi'  : {1: 8426, 2: 6783, 3: 5000, },
        		 'scale': {'f1': 1.0696e+02, 'f2': 1.7364e+01, 'f3': 9.4979e-01, }},
        'e_s2': {'Phi'  : {1: 7304, 2: 5881, 3: 5000, },
        		 'scale': {'f1': 6.4917e+02, 'f2': 1.0549e+02, 'f3': 5.7512e+00, }},
        'e_s3': {'Phi'  : {1: 8567, 2: 6943, 3: 5000, },
        		 'scale': {'f1': 1.3058e+02, 'f2': 1.7865e+01, 'f3': 1.0273e+00, }},
        'e_s4': {'Phi'  : {1: 7525, 2: 5991, 3: 5000, },
        		 'scale': {'f1': 8.0143e+02, 'f2': 1.0862e+02, 'f3': 6.2927e+00, }},
        }


class MOMF3_7_1(_MF_Base, DTLZ2):
    """Three objective functions, with 7 variables. Fidelity levels such that 
    correlation is in [0.7, 1.0] for f1, f2 and f3.
    """
    mf_def = {
        'e_r1': {'Phi'  : {1: 9168, 2: 8356, 3: 6993, 4: 5000, },
                 'scale': {'f1': 9.7701e-01, 'f2': 1.0300e+00, 'f3': 1.3075e+00, }},
        'e_r2': {'Phi'  : {1: 9989, 2: 9659, 3: 6593, 4: 5000, },
                 'scale': {'f1': 1.7128e+00, 'f2': 1.8116e+00, 'f3': 2.3048e+00, }},
        'e_r3': {'Phi'  : {1: 8647, 2: 8476, 3: 6733, 4: 5000, },
                 'scale': {'f1': 1.3973e+00, 'f2': 1.3325e+00, 'f3': 1.5736e+00, }},
        'e_s1': {'Phi'  : {1: 9318, 2: 8567, 3: 6903, 4: 5000, },
                 'scale': {'f1': 6.5451e+01, 'f2': 6.5640e+01, 'f3': 8.0547e+01, }},
        'e_s2': {'Phi'  : {1: 8967, 2: 7505, 3: 5961, 4: 5000, },
                 'scale': {'f1': 3.9860e+02, 'f2': 3.9942e+02, 'f3': 4.9223e+02, }},
        'e_s3': {'Phi'  : {1: 9599, 2: 9148, 3: 8116, 4: 5000, },
                 'scale': {'f1': 1.0640e+02, 'f2': 1.0675e+02, 'f3': 1.2502e+02, }},
        'e_s4': {'Phi'  : {1: 9058, 2: 7565, 3: 5981, 4: 5000, },
                 'scale': {'f1': 3.9885e+02, 'f2': 3.9813e+02, 'f3': 4.6518e+02, }},
        }


class MOMF3_7_2(_MF_Base, InvertedDTLZ2):
    """Three objective functions, with 7 variables. Fidelity levels such that 
    correlation is in [0.7, 1.0] for f1, f2 and f3.
    """
    mf_def = {
        'e_r1': {'Phi'  : {1: 9138, 2: 8376, 3: 6843, 4: 5000, },
                 'scale': {'f1': 1.2392e+00, 'f2': 1.2111e+00, 'f3': 9.7384e-01, }},
        'e_r2': {'Phi'  : {1: 9989, 2: 9458, 3: 6703, 4: 5000, },
                 'scale': {'f1': 2.1770e+00, 'f2': 2.1221e+00, 'f3': 1.7357e+00, }},
        'e_r3': {'Phi'  : {1: 8607, 2: 8376, 3: 6713, 4: 5000, },
                 'scale': {'f1': 1.4768e+00, 'f2': 1.5112e+00, 'f3': 1.3054e+00, }},
        'e_s1': {'Phi'  : {1: 9228, 2: 8446, 3: 6803, 4: 5000, },
                 'scale': {'f1': 7.6424e+01, 'f2': 7.6271e+01, 'f3': 6.3957e+01, }},
        'e_s2': {'Phi'  : {1: 8737, 2: 7334, 3: 5891, 4: 5000, },
                 'scale': {'f1': 4.6585e+02, 'f2': 4.6339e+02, 'f3': 3.8838e+02, }},
        'e_s3': {'Phi'  : {1: 9488, 2: 8977, 3: 7915, 4: 5000, },
                 'scale': {'f1': 1.0944e+02, 'f2': 1.0884e+02, 'f3': 9.8565e+01, }},
        'e_s4': {'Phi'  : {1: 8707, 2: 7294, 3: 5871, 4: 5000, },
                 'scale': {'f1': 4.3017e+02, 'f2': 4.2837e+02, 'f3': 3.8594e+02, }},
        }
