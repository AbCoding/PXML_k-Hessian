import numpy as np

def model_logistic_lambda(n, a, b, c, d):
    """
    As specified: y(n) = d + (a - d) / (1 + (n/c)^b)
    """
    n = np.asarray(n, dtype=float)
    return d + (a - d) / (1.0 + (n / c) ** b)

def model_power_constrained(x, a, b):
    """
    Model 2a: Power Law for Asymptote vs Divisor (n/k).
    Constraint: y(1) = 4 is enforced.
    Equation: y = a(x^b - 1) + 4
    """
    return a * (x ** b - 1) + 4

def model_power_fixed(x, a):
    """
    Model 2b: Power Law for Asymptote vs Divisor with FIXED exponent.
    Constraint: y(1) = 4 AND Fixed Exponent b = 5/3.
    Equation: y = a(x^(5/3) - 1) + 4
    """
    b_fixed = 5.0 / 3.0
    return a * (x ** b_fixed - 1) + 4