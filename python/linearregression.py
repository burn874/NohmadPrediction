#!/usr/bin/env python3

from sklearn.linear_model import LinearRegression

def linearregression(X, y):
    reg = LinearRegression().fit(X, y)
    return reg;

