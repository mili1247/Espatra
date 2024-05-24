## 1_generate_data.py

# This is the first step to use this method. It generates the target function and uses it to 
# obtain the function of the independent variable in imaginary time through forward computation.
# The function can be either fermionic or bosonic. Then, it represents the function in imaginary
# time using the coefficients of the discrete Lehmann representation (DLR), which will serve as
# the input for training.

import h5
import os
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad, trapz
from pydlr import dlr

## Parameters

# if IS_FERMIONIC is true, this code generates the A(\omega) and computes the G(\tau);
# if IS_FERMIONIC is false, this code generates the Pi(\omega) and computes the Pi(\tau).
IS_FERMIONIC = False

# Output label
output = "test"

# The size of the dataset
NB_DATA = 100

# The maximum number of peaks of A(\omega) or Pi(\omega)
NB_PICS = 7

# The number of omega points between -OMEGA_0 and OMEGA_0
NB_OMEGA = 2001

# The number of tau points between 0 and BETA
NB_TAU = 2001

# The window of energy
OMEGA_0 = 8.0

# The inverse temperature
BETA = 4.0

integral_tol = 1e-4
NOISE_LEVEL = 1e-4

###################################################################################################

## Generate Gaussian peaks
# For A(\omega), peaks are generated between [-0.4*OMEGA_0, 0.4*OMEGA_0]
# For Pi(\omega), peaks are generated between [0, 0.5*OMEGA_0]
if IS_FERMIONIC:
    wr = np.random.rand(1, NB_DATA, NB_PICS) * (0.8 * OMEGA_0) - (0.4 * OMEGA_0)
    sigma = np.random.rand(1, NB_DATA, NB_PICS) * (0.2 * OMEGA_0) + (0.02 * OMEGA_0)
else:
    wr = np.random.rand(1, NB_DATA, NB_PICS) * (0.5 * OMEGA_0) + (0. * OMEGA_0)
    sigma = np.random.rand(1, NB_DATA, NB_PICS) * (0.2 * OMEGA_0) + (0.01 * OMEGA_0)

## How much term does each data contain
R = np.random.randint(1, NB_PICS + 1, size=(NB_DATA, 1))
# Create the cancellator array
cancellator = np.ones((1, NB_DATA, NB_PICS))
for ii in range(NB_DATA):
    cancellator[0, ii, R[ii, 0]:] = 0

## Discretize the interval
omega = np.linspace(-OMEGA_0, OMEGA_0, NB_OMEGA)

## Define the sum handlers function
# Impose the bosonic target function to be odd
def A_sum_handlers(x):
    if IS_FERMIONIC:
        return np.sum(cancellator * norm.pdf(x, wr, sigma), axis=2).T
    else:
        return np.sum(cancellator * (norm.pdf(x, wr, sigma) - norm.pdf(x, -wr, sigma)), axis=2).T

## Compute A(\omega) in fermionic case or Pi(\omega) in bosonic case
A = A_sum_handlers(omega)
# Normalize A
if IS_FERMIONIC:
    NORMALIZATION_FACTOR = trapz(A, omega, axis=1)
else:
    NORMALIZATION_FACTOR = trapz(A[:, (NB_OMEGA + 1) // 2:], omega[(NB_OMEGA + 1) // 2:], axis=1)
A = A / NORMALIZATION_FACTOR[:, np.newaxis]

## The Pi(\omega) are not necessarily normalized. Therefore a random factor is introduced
if not IS_FERMIONIC:
    RANDOM_FACTOR = np.random.rand(NB_DATA, 1) * 4 - 2
    A = A * RANDOM_FACTOR

## Define the kernel handler function
def kernel_handler(tau, x):
    if IS_FERMIONIC:
        x = x[:, np.newaxis]
        expr1 = (x * (BETA - tau) >= 37) * np.exp(-x * tau) / (1 + np.exp(-x * BETA))
        expr2 = (x * (BETA - tau) < 37) * np.exp(x * (BETA - tau)) / (np.exp(BETA * x) + 1)
        return expr1 + expr2
    else:
        return np.cosh(x[:, np.newaxis] * (tau - BETA / 2)) / np.sinh(x[:, np.newaxis] * BETA / 2)

## Define the G handlers function
def G_handlers(tau):
    integrand = lambda x: kernel_handler(tau, x).T @ A_sum_handlers(x).T
    if IS_FERMIONIC:
        integral_result = np.array([quad(integrand, -OMEGA_0, OMEGA_0, epsabs=integral_tol)[0] for tau_i in tau])
        return np.random.normal(loc=integral_result / NORMALIZATION_FACTOR[:, np.newaxis], scale=NOISE_LEVEL)
    else:
        integral_result = np.array([quad(integrand, 0, OMEGA_0, epsabs=integral_tol)[0] for tau_i in tau])
        return np.random.normal(loc=integral_result / (NORMALIZATION_FACTOR * np.pi)[:, np.newaxis] * RANDOM_FACTOR[:, np.newaxis], scale=NOISE_LEVEL)

taus = np.linspace(0, BETA, NB_TAU)
G = G_handlers(taus).T
