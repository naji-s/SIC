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


# entering the series of X and Y below for finding the causal direction with confidence intervals with null hypothesis
# of rotation invariant prior
print noisy_SDR_estimator(X, Y, p_value_type = 'rotation', calc_CI = True, p_value_sample_size = 1000, ic_type = 'mdl',
                          ic_max_order = 20)
print noisy_SDR_estimator(Y, X, p_value_type = 'rotation', calc_CI = True, p_value_sample_size = 1000, ic_type = 'mdl',
                          ic_max_order = 20)


# calculating SDR for (multivariate/singlevariate) time series with possible whitening.

# definign the first filter
A_1 = np.random.randn(FO)
X_1 = np.random.randn(x_size)
Y_1 = signal.lfilter(A_1, [1.], X_1)

# defining the secodn filter
A_2 = np.random.randn(FO)
X_2 = np.random.randn(x_size)
Y_2 = signal.lfilter(A, [1.], X_2)

# stackinjg the series to make two-variate time series
X = np.vstack((X_1, X_2))
Y = np.vstack((Y_1, Y_2))

# reporting the results without whitening
print "Forward SDRs without whitening:\n", noiseless_SDR_estimator(X, Y, whitening=False)
print "Backward SDRs without whitening:\n", noiseless_SDR_estimator(Y, X, whitening=False)

# reporting the results with whitening
print "Forward SDRs with whitening:\n", noiseless_SDR_estimator(X, Y, whitening=True)
print "Backward SDRs with whitening:\n", noiseless_SDR_estimator(Y, X, whitening=True)
