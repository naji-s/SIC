""" The main code for SIC algorithm."""

from numpy import random as random
from scipy import signal
from SIC_toolkit import SIC_inference, stochastic_SDR_estimator, deterministic_SDR_estimator


###### Illustration of an example ########
##
##
##
FO = 11
x_size = 5000
A = np.random.randn(FO)


# calculating SDR for two univariate time series.

# defining the first filter and its output as the cause time series
A_1 = random.randn(FO)
Z = random.randn(x_size)
X = signal.lfilter(A_1, [1.], Z)
f = file('/Users/naji/SVN/Papers/Spectral_Independence_Criterion_SIC/Codes/X_file.txt', 'w')
for i in range(X.shape[0]):
    f.write(str(X[i])+'\n')
f.close()
# defining the secodn filter which take the cause as input and generates the effect
A_2 = random.randn(FO)
Y = signal.lfilter(A, [1.], X)
f = file('/Users/naji/SVN/Papers/Spectral_Independence_Criterion_SIC/Codes/Y_file.txt', 'w')
for i in range(Y.shape[0]):
    f.write(str(Y[i])+'\n')
f.close()

# the inference step using algorithm 1 in ICML 2015
SIC_inference(X, Y)