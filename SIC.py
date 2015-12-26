""" The main code for SIC algorithm."""

import numpy as np
from scipy import signal
from SIC_toolkit import SIC_inference, stochastic_SDR_estimator, deterministic_SDR_estimator


###### Illustration of an example ########
##
##
##
FO = 11
x_size = 5000
A = np.random.randn(FO)
#print np.linalg.norm(A)**2
#input X as random noise to the mechanism
X = np.random.randn(x_size)
# setting up an IIR filter with coeffcieints A and white noise as input,
# to generate the output Y
Y = signal.lfilter(A, [1.], X)
Y += np.random.randn(Y.shape[0]) * .12


# calculating SDR for two univariate time serie.

# defining the first filter and its output as the cause time series
A_1 = np.random.randn(FO)
Z = np.random.randn(x_size)
X = signal.lfilter(A_1, [1.], Z)

# defining the secodn filter which take the cause as input and generates the effect
A_2 = np.random.randn(FO)
Y = signal.lfilter(A, [1.], X)

# the inference step using algorithm 1 in ICML 2015
SIC_inference(X, Y)