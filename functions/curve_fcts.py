import numpy as np
from scipy.stats import gamma
from scipy.stats import johnsonsu


# collection of different functions

# define functions
def Budyko_curve(aridity, **kwargs):
    # Budyko, M.I., Miller, D.H. and Miller, D.H., 1974. Climate and life (Vol. 508). New York: Academic press.
    return np.sqrt(aridity * np.tanh(1 / aridity) * (1 - np.exp(-aridity)));


def Berghuijs_recharge_curve(aridity):
    alpha = 0.72
    beta = 15.11
    RR = alpha * (1 - (np.log(aridity ** beta + 1) / (1 + np.log(aridity ** beta + 1))))
    return RR


def Berghuijs_recharge_curve_alt(aridity):
    alpha = 0.72
    beta = 15.11
    gamma = 0.2
    RR = np.exp(-gamma / aridity) * alpha * (1 - (np.log(aridity ** beta + 1) / (1 + np.log(aridity ** beta + 1))))
    return RR


def alt_recharge_curve(x, alpha, beta, gamma):
    # alpha = 0.72
    # beta = 15.11
    RR = np.exp(-gamma / x) * alpha * (1 - (np.log(x ** beta + 1) / (1 + np.log(x ** beta + 1))))
    return RR


def fast_flow_curve(P, Q, Smax, **kwargs):
    # Smax = max storage
    Qf = P - Smax
    Qf[Qf < 0] = 0
    Qf[Qf > Q] = Q[Qf > Q]
    return Qf


def lognormal_pdf(x, mu, sigma, scale):
    return scale * (1 / (x * sigma * np.sqrt(2 * np.pi)) * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2)))


def gamma_pdf(x, shape, scale):
    return (gamma.pdf(x, a=shape, scale=scale))


def loggamma_pdf(x, shape, loc, scale):
    return gamma.pdf(x, a=shape, loc=loc, scale=scale)


def johnsonsu_distribution(x, gamma, delta, xi, lam):
    return johnsonsu.pdf(x, a=gamma, b=delta, loc=xi, scale=lam)


def Gabs_function(x, c, k, l):
    return c * (1 - np.exp(-k * x)) * np.exp(-l * (x - 1))
