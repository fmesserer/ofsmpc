import numpy as np
import casadi as ca

def huberLoss(x, sigma=1):
    return np.sqrt( x**2 + sigma**2  ) - sigma


def pdf_stdn(x):
    return 1 / ca.sqrt(2 * ca.pi) * ca.exp(-.5 * x**2)


def cdf_stdn(x):
    return .5 * ( 1 + ca.erf(x / ca.sqrt(2)) )


def expectation_over_relu(mu, sigma):
    frac = mu / sigma
    return sigma * pdf_stdn(frac) + mu * cdf_stdn(frac)
