"""This file contains the toolkits for using SIC"""
# import numpy as np
from scipy.signal import welch, lfilter
from statsmodels.regression.linear_model import OLS
from itertools import product
from numpy import mean, multiply, \
    empty, ones, divide, einsum, sqrt, float64, abs, log, append,roots, asarray,arange, tile, var

from numpy import any as _any

def cov(a, b):
    return mean(nmultiply((a-mean(a)),(b-mean(b))))

def sphere_sampling(dim = 1, sample_size = 1):
    X_n = random.randn(sample_size * dim).reshape((sample_size, dim))
    X_n_norm = sqrt(einsum('ij, ij -> i', X_n, X_n))
    return X_n / X_n_norm[..., None].astype(float64)


def p_value_calculator(X, x, max_S, power, log_scale=False):
    if log_scale:
        X = log(X)
        x = log(x)
        abs_mean_distance = abs(X)
        abs_x_distance = abs(x)
    else:
        X_mean = mean(X)
        abs_mean_distance = abs(X - X_mean)
        abs_x_distance = abs(x - X_mean)
    # plt.hist(X, bins = 50)
    # plt.show()
    return mean(abs_x_distance < abs_mean_distance)#* (max_S)/float(power))


# globalizing and defaulting the boolean p_mat_flag which indiactes wheter the 
# preprocessing for rotation (translation) matrix for rotation invariant 
# (translation invariant) prior in time domain for IR function (in frequency 
# domain for transfer function

def calc_ref_measure(X):
    freq, S_x_s = welch(X)
    return mean(X, axis = 0)
def SIC_for_IIR(FO=11, BO=3, AR_amp = 0.1, ts_size = 5000, sigma_input_noise=0., sigma_output_noise=0.
                , order=10, report_CI=False, ic_type=None, ic_max_order=None,welch_window_size=1000):
    """A function to assess the performance of SIC under additive noise on input, output, both and under
        denoising in all the previous scenarios"""

    print sigma_input_noise, sigma_output_noise

    # a coefficient to reduce the influence of autoregressive coefficients to increase the chance of stability
    AR_amp_x = AR_amp_z = AR_amp

    # setting the filter coefficients that generates X_t
    A_z = append(1.,random.randn(FO))
    B_z = append(1.,random.randn(BO) * AR_amp_z)

    #checking whether the first filter has coefficients give rise to a stable IIR filter
    while any(abs(roots(A_z)) >= 0.95) or _any(abs(roots(B_z)) >= 0.95):
        A_z = append(1.,random.randn(FO))
        B_z = append(1.,random.randn(BO) * AR_amp_z)


    # setting the filter coefficients that generates Y_t
    A_x = append(1.,random.randn(FO))
    B_x = append(1.,random.randn(BO) * AR_amp_x)

    #checking whether the second filter has coefficients give rise to a stable IIR filter
    while any(abs(roots(A_x)) >= 0.95) or any(abs(roots(B_x)) >= 0.95):
        A_x = append(1.,random.randn(FO))
        B_x = append(1.,random.randn(BO) * AR_amp_x)


    #input Z as random noise to a mechanism that generates the cause: X_t
    Z = random.randn(ts_size)
    X = lfilter(A_z, B_z, Z)

    # adding additive noise to the cause
    noisy_X = X + random.randn(X.shape[0]) * sigma_input_noise

    Y = lfilter(A_x, B_x, X)

    # adding additive noise to the effect
    noisy_Y = Y + random.randn(Y.shape[0]) * sigma_output_noise

    return_SDR_dict = dict()

    return_SDR_dict['stochastic_in_out'] = asarray([stochastic_SDR_estimator(noisy_X, noisy_Y, report_CI=report_CI, ic_max_order=ic_max_order,
                                        ic_type=ic_type), stochastic_SDR_estimator(noisy_Y, noisy_X, report_CI=report_CI,
                                        ic_max_order=ic_max_order, ic_type=ic_type)])
    if sigma_input_noise > 0:
        return_SDR_dict['stochastic_in'] = asarray([stochastic_SDR_estimator(noisy_X, Y, report_CI=report_CI, ic_max_order=ic_max_order,
                                        ic_type=ic_type), stochastic_SDR_estimator(Y, noisy_X, report_CI=report_CI,
                                        ic_max_order=ic_max_order, ic_type=ic_type)])
    else:
        return_SDR_dict['stochastic_in'] = return_SDR_dict['stochastic_in_out']
    if sigma_output_noise > 0:
        return_SDR_dict['stochastic_out'] = asarray([stochastic_SDR_estimator(X, noisy_Y, report_CI=report_CI, ic_max_order=ic_max_order,
                                        ic_type=ic_type), stochastic_SDR_estimator(noisy_Y, X, report_CI=report_CI,
                                        ic_max_order=ic_max_order, ic_type=ic_type)])
    else:
        return_SDR_dict['stochastic_out'] = return_SDR_dict['stochastic_in_out']

    return_SDR_dict['deterministic_in_out'] = asarray([deterministic_SDR_estimator(noisy_X, noisy_Y,
                                                                           welch_window_size=welch_window_size),
                                               deterministic_SDR_estimator(noisy_Y, noisy_X,
                                                                           welch_window_size=welch_window_size)])

    return_SDR_dict['deterministic_in'] = asarray([deterministic_SDR_estimator(noisy_X, Y, welch_window_size=welch_window_size),
                                           deterministic_SDR_estimator(Y,noisy_X, welch_window_size=welch_window_size)])

    return_SDR_dict['deterministic_out'] = asarray([deterministic_SDR_estimator(X, noisy_Y, welch_window_size=welch_window_size),
                                            deterministic_SDR_estimator(noisy_Y, X, welch_window_size=welch_window_size)])
    return var(X), var(Y), return_SDR_dict


def deterministic_SDR_estimator(X, Y, ref_measure = None,whitening = False, welch_window_size=500):
    # if X.shape[0] < 2:
    #     print "Whitening in input domain not possible! The input time series is univariate"
    # if Y.shape[0] < 2:
    #     print "Whitening in output domain not possible! The output time series is univariate"
    if len(X.shape) == 1:
        X = X[None,...]
    if len(Y.shape) == 1:
        Y = Y[None,...]
    rho_s = empty((X.shape[0], Y.shape[0]))
    freqs, S_x_s = welch(X, nperseg = welch_window_size, return_onesided=False)
    freqs, S_y_s = welch(Y, nperseg = welch_window_size, return_onesided=False)
    if whitening:
        if ref_measure:
            mean_S_x = ref_measure
            mean_S_y = ref_measure
        else:
            mean_S_x = mean(S_x_s, axis = 0)
            mean_S_y = mean(S_y_s, axis = 0)
    else:
        mean_S_x = ones(S_x_s.shape[-1])
        mean_S_y = ones(S_y_s.shape[-1])

    new_S_x_s = divide(S_x_s, mean_S_x)
    new_S_y_s = divide(S_y_s, mean_S_y)
    for x_ind, y_ind in product(range(new_S_x_s.shape[0]), range(new_S_y_s.shape[0])):
        abs_h_square = divide(new_S_y_s[y_ind], new_S_x_s[x_ind])

        h_square_mean = mean(abs_h_square)
        C_X_0 = mean(new_S_x_s[x_ind])
        C_Y_0 = mean(new_S_y_s[y_ind])
        rho_s[x_ind, y_ind] = float(C_Y_0 ) /(float(C_X_0) * float(h_square_mean))
    return rho_s.squeeze()


def stochastic_SDR_estimator(X, Y, welch_window_size=256, report_CI=True, p_value_sample_size=2000, p_value_type='rotation',
                        fitting_order=None, ic_type=None, ic_max_order=None):
    """ function stochastic_SDR_estimator will take input and output time series and calculates the
    SDR from X to Y where it is assumed that the filter is FIR and there is additive noise in the aoutput of the filter.
     The spectral densities in the expression for SDR are calculated using Welch's method, but this can be replaced with
     any preferred method of spectral density estimation


    inputs:
        X (ndarray(float64)):    1D input time series array
        Y (ndarray(float64)):    1D output time series array
        nperseg (int):           size of the window for Welch's method

    outputs:
        rho_X_Y (float64):    SDR value \rho_{X\to Y}
    """
    # check whether the variables are zero meaned:

    X_mean = mean(X)
    Y_mean = mean(Y)
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
    h_norm = linalg.norm(h_fit)
    C_X_0 = var(X)
    C_Y_0 = var(Y)

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
    #     h_norm = sqrt(mean(h_hat_squared))
    #     C_X_0 = mean(Sx)
    #     C_Y_0 = mean(Sy)


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
            y_s = asarray(y_s)
            p_val_Cy_0_mat = var(y_s, axis = -1)
        freqs, Sx = welch(X, nperseg=welch_window_size, return_onesided=False)
        if p_value_type is 'translation':
            _, Sy = welch(Y, nperseg=welch_window_size, return_onesided=False)
            num_freqs = freqs.shape[0]
            h_hat_squared = Sy / Sx
            idx_1 = tile(arange(num_freqs), (num_freqs, 1))
            idx_2 = idx_1.T
            p_val_h_matrix = h_hat_squared[idx_1 - idx_2]
            p_val_Cy_0_mat = mean(multiply(p_val_h_matrix, Sx[None, ...]), axis = 1)
        p_val_rho_mat = p_val_Cy_0_mat / (float(C_X_0) * float(h_norm ** 2))
        max_S_x = max(Sx)
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
        return "X_t causes Y_t"
    else:
        return "Y_t causes X_t"


def win_sig(x, seg_size):
    """
    A function just to cut a 1D time series into pieces of specific length (seg_size)
    input:
        x (ndarray64):   the input 1D time series
        seg_size:            the segment size that is used to split the time series and
        concatenate the split parts in the other dimension
    output:
        rearranged time series
        
    Example:
        input:
            x = asarray([1,2,3,4,5,6])
        output:
            [[1 2 3 4 5]
            [2 3 4 5 6]]

    """
    #index set manipulation for generating the split version of the signals faster
    sig_len = x.shape[0] - seg_size + 1
    idx_temp = indices((sig_len, seg_size))
    idx_temp = idx_temp[0] + idx_temp[1]
    return x[idx_temp].reshape((-1, seg_size))


def mdl(OLS_result):
    return -2 * OLS_result.llf + 1./2 * OLS_result.df_model * log(OLS_result.df_resid)


def C_FIR_fit(x, y, fitting_order=50, ic_type=None, ic_max_order=None):
    """This function takes 1D time series x and 1D time series y and tries to regress
    y with x linearly
    input:
        x:                ndarray64: input time series
        y:                ndarray64: output time series
        order:            int64: order of the fitting
        trend_order:      int: trend order as it is defined for utils.py for statsmodels see:
                          https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/vector_ar/util.py
                          for mor details.
    output:
        w:                int64: the linear coefficients
    """

    ic = dict()
    ic['val'] = inf
    ic['ord'] = 1
    if fitting_order is None:
        if ic_type is None or ic_max_order is None:
            # print "No fitting order or IC specified. Setting the IC to aic and max order to 20..."
            ic_max_order = 20
            ic_type = 'aic'
        for o in arange(1, ic_max_order):
            # print "aic:",aic
            x_mat = win_sig(x, o)
            y_mat = y[o - 1:]
            # resid_size = y_mat.shape[0]
            # w, squared_sum= linalg.lstsq(x_mat, y_mat)[0:2]
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
        w = linalg.lstsq(x_mat, y_mat)[0]
    return w, ic