"""This file contains the toolkits for using SIC"""
import numpy as np
from scipy.signal import welch, lfilter
from scipy.fftpack import rfft, fftfreq
from statsmodels.tools.eval_measures import aic as eval_aic
from statsmodels.regression.linear_model import OLS
#from scipy.stats.mstats import mode
from matplotlib import pyplot as plt
from itertools import product
from scipy.signal import lfilter


def cov(a, b):
    return np.mean(np.multiply((a-np.mean(a)),(b-np.mean(b))))

def sphere_sampling(dim = 1, sample_size = 1):
    X_n = np.random.randn(sample_size * dim).reshape((sample_size, dim))
    X_n_norm = np.sqrt(np.einsum('ij, ij -> i', X_n, X_n))
    return X_n / X_n_norm[..., None].astype(np.float64)


def p_value_calculator(X, x, max_S, power, log_scale=False):
    if log_scale:
        X = np.log(X)
        x = np.log(x)
        abs_mean_distance = np.abs(X)
        abs_x_distance = np.abs(x)
    else:
        X_mean = np.mean(X)
        abs_mean_distance = np.abs(X - X_mean)
        abs_x_distance = np.abs(x - X_mean)
    # plt.hist(X, bins = 50)
    # plt.show()
    return np.mean(abs_x_distance < abs_mean_distance)#* (max_S)/float(power))


# globalizing and defaulting the boolean p_mat_flag which indiactes wheter the 
# preprocessing for rotation (translation) matrix for rotation invariant 
# (translation invariant) prior in time domain for IR function (in frequency 
# domain for transfer function

def calc_ref_measure(X):
    freq, S_x_s = welch(X)
    return np.mean(X, axis = 0)
def SIC_for_IIR(FO=11, BO=3, AR_amp = 0.1, ts_size = 5000, sigma_input_noise=0., sigma_output_noise=0.
                , order=10, report_CI=False, ic_type=None, ic_max_order=None,welch_window_size=1000):
    """A function to assess the performance of SIC under additive noise on input, output, both and under
        denoising in all the previous scenarios"""

    print sigma_input_noise, sigma_output_noise

    # a coefficient to reduce the influence of autoregressive coefficients to increase the chance of stability
    AR_amp_x = AR_amp_z = AR_amp

    # setting the filter coefficients that generates X_t
    A_z = np.append(1.,np.random.randn(FO))
    B_z = np.append(1.,np.random.randn(BO) * AR_amp_z)

    #checking whether the first filter has coefficients give rise to a stable IIR filter
    while np.any(np.abs(np.roots(A_z)) >= 0.95) or np.any(np.abs(np.roots(B_z)) >= 0.95):
        A_z = np.append(1.,np.random.randn(FO))
        B_z = np.append(1.,np.random.randn(BO) * AR_amp_z)


    # setting the filter coefficients that generates Y_t
    A_x = np.append(1.,np.random.randn(FO))
    B_x = np.append(1.,np.random.randn(BO) * AR_amp_x)

    #checking whether the second filter has coefficients give rise to a stable IIR filter
    while np.any(np.abs(np.roots(A_x)) >= 0.95) or np.any(np.abs(np.roots(B_x)) >= 0.95):
        A_x = np.append(1.,np.random.randn(FO))
        B_x = np.append(1.,np.random.randn(BO) * AR_amp_x)


    #input Z as random noise to a mechanism that generates the cause: X_t
    Z = np.random.randn(ts_size)
    X = lfilter(A_z, B_z, Z)

    # adding additive noise to the cause
    noisy_X = X + np.random.randn(X.shape[0]) * sigma_input_noise

    Y = lfilter(A_x, B_x, X)

    # adding additive noise to the effect
    noisy_Y = Y + np.random.randn(Y.shape[0]) * sigma_output_noise

    return_SDR_dict = dict()

    return_SDR_dict['stochastic_in_out'] = np.asarray([stochastic_SDR_estimator(noisy_X, noisy_Y, report_CI=report_CI, ic_max_order=ic_max_order,
                                        ic_type=ic_type), stochastic_SDR_estimator(noisy_Y, noisy_X, report_CI=report_CI,
                                        ic_max_order=ic_max_order, ic_type=ic_type)])
    if sigma_input_noise > 0:
        return_SDR_dict['stochastic_in'] = np.asarray([stochastic_SDR_estimator(noisy_X, Y, report_CI=report_CI, ic_max_order=ic_max_order,
                                        ic_type=ic_type), stochastic_SDR_estimator(Y, noisy_X, report_CI=report_CI,
                                        ic_max_order=ic_max_order, ic_type=ic_type)])
    else:
        return_SDR_dict['stochastic_in'] = return_SDR_dict['stochastic_in_out']
    if sigma_output_noise > 0:
        return_SDR_dict['stochastic_out'] = np.asarray([stochastic_SDR_estimator(X, noisy_Y, report_CI=report_CI, ic_max_order=ic_max_order,
                                        ic_type=ic_type), stochastic_SDR_estimator(noisy_Y, X, report_CI=report_CI,
                                        ic_max_order=ic_max_order, ic_type=ic_type)])
    else:
        return_SDR_dict['stochastic_out'] = return_SDR_dict['stochastic_in_out']

    return_SDR_dict['deterministic_in_out'] = np.asarray([deterministic_SDR_estimator(noisy_X, noisy_Y,
                                                                           welch_window_size=welch_window_size),
                                               deterministic_SDR_estimator(noisy_Y, noisy_X,
                                                                           welch_window_size=welch_window_size)])

    return_SDR_dict['deterministic_in'] = np.asarray([deterministic_SDR_estimator(noisy_X, Y, welch_window_size=welch_window_size),
                                           deterministic_SDR_estimator(Y,noisy_X, welch_window_size=welch_window_size)])

    return_SDR_dict['deterministic_out'] = np.asarray([deterministic_SDR_estimator(X, noisy_Y, welch_window_size=welch_window_size),
                                            deterministic_SDR_estimator(noisy_Y, X, welch_window_size=welch_window_size)])
    return np.var(X), np.var(Y), return_SDR_dict


def deterministic_SDR_estimator(X, Y, ref_measure = None,whitening = False, welch_window_size=500):
    # if X.shape[0] < 2:
    #     print "Whitening in input domain not possible! The input time series is univariate"
    # if Y.shape[0] < 2:
    #     print "Whitening in output domain not possible! The output time series is univariate"
    if len(X.shape) == 1:
        X = X[None,...]
    if len(Y.shape) == 1:
        Y = Y[None,...]
    rho_s = np.empty((X.shape[0], Y.shape[0]))
    freqs, S_x_s = welch(X, nperseg = welch_window_size, return_onesided=False)
    freqs, S_y_s = welch(Y, nperseg = welch_window_size, return_onesided=False)
    if whitening:
        if ref_measure:
            mean_S_x = ref_measure
            mean_S_y = ref_measure
        else:
            mean_S_x = np.mean(S_x_s, axis = 0)
            mean_S_y = np.mean(S_y_s, axis = 0)
    else:
        mean_S_x = np.ones(S_x_s.shape[-1])
        mean_S_y = np.ones(S_y_s.shape[-1])

    new_S_x_s = np.divide(S_x_s, mean_S_x)
    new_S_y_s = np.divide(S_y_s, mean_S_y)
    for x_ind, y_ind in product(range(new_S_x_s.shape[0]), range(new_S_y_s.shape[0])):
        abs_h_square = np.divide(new_S_y_s[y_ind], new_S_x_s[x_ind])

        h_square_mean = np.mean(abs_h_square)
        C_X_0 = np.mean(new_S_x_s[x_ind])
        C_Y_0 = np.mean(new_S_y_s[y_ind])
        rho_s[x_ind, y_ind] = float(C_Y_0 ) /(float(C_X_0) * float(h_square_mean))
    return rho_s.squeeze()


def stochastic_SDR_estimator(X, Y, welch_window_size=256, report_CI=True, p_value_sample_size=2000, p_value_type='rotation',
                        fitting_order=None, ic_type=None, ic_max_order=None):
    """ function stochastic_SDR_estimator will take input and output time series and calculates the
    SDR from X to Y where it is assumed that the filter is FIR and there is additive noise in the aoutput of the filter.
     The spectral densities in the expression for SDR are calculated using Welch's method, but this can be replaced with
     any preferred method of spectral density estimation


    inputs:
        X (np.ndarray(float64)):    1D input time series array
        Y (np.ndarray(float64)):    1D output time series array
        nperseg (np.int):           size of the window for Welch's method

    outputs:
        rho_X_Y (np.float64):    SDR value \rho_{X\to Y}
    """
    # check whether the variables are zero meaned:

    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    if X_mean != 0:
        # print "The mean of X is not zero. Setting it to zero..."
        X = X - X_mean
    if Y_mean != 0:
        # print "The mean of Y is not zero. Setting it to zero..."
        Y = Y - Y_mean
    h_fit, ic = C_FIR_fit(X, Y, fitting_order=fitting_order, ic_type=ic_type, ic_max_order=ic_max_order)
    if ic is not None:
        fitting_order = ic['ord']
    Y = lfilter(h_fit, [1.], X)
    h_norm = np.linalg.norm(h_fit)
    C_X_0 = np.var(X)
    C_Y_0 = np.var(Y)

    #################################################################################
    # old section to calculate deterministic SDR estimates which has its own function now
    #################################################################################
    # if  (denoising_params is not None) or denoising_params['order'] == None:
    #     if CI['p_value_type'] is None:
    #         print "There is no denosing taking place. It is not possible to calculate the p value. Setting p-value type to None"
    #         p_value_type = None
    #     freqs, Sx = welch(X, nperseg = nperseg, return_onesided = False)
    #     _, Sy = welch(Y, nperseg = nperseg, return_onesided = False)
    #     num_freqs = freqs.shape[0]
    #     h_hat_squared = Sy / Sx
    #     h_norm = np.sqrt(np.mean(h_hat_squared))
    #     C_X_0 = np.mean(Sx)
    #     C_Y_0 = np.mean(Sy)


    #################################################################################
    # calculating the real SDR value with input X and output Y
    #################################################################################

    rho_X_Y = C_Y_0 / (float(C_X_0) * float(h_norm ** 2))
    if report_CI:
        if p_value_type is 'rotation':
            h_coef_matrix = sphere_sampling(fitting_order, p_value_sample_size) * h_norm
            y_s = []
            for i in range(p_value_sample_size):
                y_s.append(lfilter(h_coef_matrix[i], [1.], X))
            y_s = np.asarray(y_s)
            p_val_Cy_0_mat = np.var(y_s, axis = -1)
        freqs, Sx = welch(X, nperseg=welch_window_size, return_onesided=False)
        if p_value_type is 'translation':
            _, Sy = welch(Y, nperseg=welch_window_size, return_onesided=False)
            num_freqs = freqs.shape[0]
            h_hat_squared = Sy / Sx
            idx_1 = np.tile(np.arange(num_freqs), (num_freqs, 1))
            idx_2 = idx_1.T
            p_val_h_matrix = h_hat_squared[idx_1 - idx_2]
            p_val_Cy_0_mat = np.mean(np.multiply(p_val_h_matrix, Sx[None, ...]), axis = 1)
        p_val_rho_mat = p_val_Cy_0_mat / (float(C_X_0) * float(h_norm ** 2))
        max_S_x = np.max(Sx)
        p_val = p_value_calculator(p_val_rho_mat, rho_X_Y, max_S=max_S_x, power=C_X_0)
    else:
        p_val = None
    return rho_X_Y, p_val

                
def SIC_inference(X, Y, stochastic=False, order=None):
    """Function that takes SDRs in both directions and compares them. This will
    later be replaced with a more advanced report of the causal direction with confidence
    intervals"""

    if stochastic:
        if order is None:
            rho_X_Y, p_X_Y = stochastic_SDR_estimator(X, Y, ic_type='bic', ic_max_order=20)
            rho_Y_X, p_Y_X = stochastic_SDR_estimator(Y, X, ic_type='bic', ic_max_order=20)
        else:
            rho_X_Y, p_X_Y = stochastic_SDR_estimator(X, Y, order=order)
            rho_Y_X, p_Y_X = stochastic_SDR_estimator(Y, X, order=order)
    else:
        rho_X_Y = deterministic_SDR_estimator(X, Y)
        rho_Y_X = deterministic_SDR_estimator(Y, X)

    if rho_X_Y > rho_Y_X:
        print "X_t causes Y_t"
    else:
        print "Y_t causes X_t"


def win_sig(x, seg_size):
    """
    A function just to cut a 1D time series into pieces of specific length (seg_size)
    input:
        x (np.ndarray64):   the input 1D time series
        seg_size:            the segment size that is used to split the time series and
        concatenate the split parts in the other dimension
    output:
        rearranged time series
        
    Example:
        input:
            x = np.asarray([1,2,3,4,5,6])
        output:
            [[1 2 3 4 5]
            [2 3 4 5 6]]

    """
    #index set manipulation for generating the split version of the signals faster
    sig_len = x.shape[0] - seg_size + 1
    idx_temp = np.indices((sig_len, seg_size))
    idx_temp = idx_temp[0] + idx_temp[1]
    return x[idx_temp].reshape((-1, seg_size))


def mdl(OLS_result):
    return -2 * OLS_result.llf + 1./2 * OLS_result.df_model * np.log(OLS_result.df_resid)


def C_FIR_fit(x, y, fitting_order=50, ic_type=None, ic_max_order=None):
    """This function takes 1D time series x and 1D time series y and tries to regress
    y with x linearly
    input:
        x:                np.ndarray64: input time series
        y:                np.ndarray64: output time series
        order:            np.int64: order of the fitting
        trend_order:      int: trend order as it is defined for utils.py for statsmodels see:
                          https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/vector_ar/util.py
                          for mor details.
    output:
        w:                np.int64: the linear coefficients
    """

    ic = dict()
    ic['val'] = np.inf
    ic['ord'] = 1
    if fitting_order is None:
        if ic_type is None or ic_max_order is None:
            # print "No fitting order or IC specified. Setting the IC to aic and max order to 20..."
            ic_max_order = 20
            ic_type = 'aic'
        for o in np.arange(1, ic_max_order):
            # print "aic:",aic
            x_mat = win_sig(x, o)
            y_mat = y[o - 1:]
            # resid_size = y_mat.shape[0]
            # w, squared_sum= np.linalg.lstsq(x_mat, y_mat)[0:2]
            temp_OLS = OLS(y_mat, x_mat).fit()
            if ic_type == 'aic':
                new_ic = temp_OLS.aic
            elif ic_type == 'bic':
                new_ic = temp_OLS.bic
            elif ic_type == 'mdl':
                new_ic = mdl(temp_OLS)

            if ic['val'] > new_ic:
                ic['val'] = new_ic
                ic['ord'] = o
        print "final",ic_type,":", ic
        return temp_OLS.params, ic
    else:
        x_mat = win_sig(x,fitting_order)
        y_mat = y[fitting_order - 1:]
        w = np.linalg.lstsq(x_mat, y_mat)[0]
    return w, ic