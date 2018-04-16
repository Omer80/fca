#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 16:41:54 2018

@author: ohm
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("fast_convolution_integral_algorithm_cy.pyx", annotate=True),
)
