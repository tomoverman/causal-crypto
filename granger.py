import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class granger():
    ### INPUTS ###
    ## coeffs: matrix of the coefficients used in the AR model
    ##         if inconsistent number of lagged vals used for certain variables,
    ##		   just add zeros. The first entry in each row is the constant term, then
    ##		   are the coefficients of all of the lagged first variables, then the lagged
    ##		   second variables, and so on...
    ## num_vars: number of variables used in the multivariate model
    ## N: total number of time steps for generating data
    ## std_devs: vector of standard deviations for noise component of each var
    ## init_vals: matrix of initial values, must be consistent with coeff matrix
    def __init__(self, data):
        self.data=data
        self.num_vars=data.shape[0]

    # check that the sizes of the various inputs are consistent
    def _check_consistency(self):
        num_vars = self.num_vars
        coeffs = self.coeffs
        std_devs = self.std_devs
        init_vals = self.init_vals
        # check that number of variables is correct in coeff matrix
        assert coeffs.shape[0] == num_vars, "Number of rows in coeff =/= num_vars"

        # check that number of variables is correct in std_devs
        assert std_devs.shape[0] == num_vars, "Length of std_devs =/= num_vars"

        # check that number of variables is correct in init_vals
        assert init_vals.shape[0] == num_vars, "Number of rows in init_vals =/= num_vars"

        # check that enough initial values are provided
        assert init_vals.shape[1] == coeffs.shape[1] // num_vars, "Number of initial values provided not correct"

    # generates model fits so we can get the distribution of residuals
    # inputs:
    #           variable_to_fit -- response variable in regression
    #           variable_to_test -- variable that you are testing to see if it causes variable_to_fit
    #           p -- number of lags to consider
    def _generate_residual_dists(self, variable_to_fit, variable_to_test, p):
        hist = self.data
        # get lagged values
        hist_lag = np.roll(hist, 1, 1)

        # get total number of variables
        dimension = self.num_vars

        # get a list of all variables (including lags) except the one you are testing for causality
        slice_reduced = list(range(dimension * p))
        for i in range(p):
            slice_reduced.pop(variable_to_test + i * dimension - i)
        # print(slice_reduced)

        data_in = hist_lag[:, p:].T
        for i in range(p - 1):
            hist_lag2 = np.roll(hist_lag, i + 1, 1)
            data_in = np.concatenate((data_in, hist_lag2[:, p:].T), axis=1)

        # fit full model
        full_reg = LinearRegression().fit(data_in, hist[variable_to_fit, p:])

        # fit reduced model
        reduced_reg = LinearRegression().fit(data_in[:, slice_reduced], hist[variable_to_fit, p:])

        # get predictions on data
        full_prediction = full_reg.predict(data_in)
        reduced_prediction = reduced_reg.predict(data_in[:, slice_reduced])

        # get residuals
        full_residuals = full_prediction - hist[variable_to_fit, p:]
        reduced_residuals = reduced_prediction - hist[variable_to_fit, p:]

        # return residuals
        return full_residuals, reduced_residuals

    def causality(self, order, alpha=0.05):
        print("\\begin{tabular}{cccc}")
        print("\hline")
        print(r"Response Variable & Causal Variable & $p$-value & Causality?\\")
        print("\hline")
        for variable_to_fit in range(self.num_vars):
            for variable_to_test in range(self.num_vars):
                if variable_to_fit != variable_to_test:
                    [full_res, reduced_res] = self._generate_residual_dists(variable_to_fit, variable_to_test, order)
                    full_std = np.std(full_res)
                    reduced_std = np.std(reduced_res)
                    F_star = reduced_std * reduced_std / full_std / full_std
                    p = 1 - scipy.stats.f.cdf(F_star, len(full_res) - 1, len(reduced_res) - 1)
                    # print("Testing to see if variable {:d} causes variable {:d}:".format(variable_to_test, variable_to_fit), p < alpha)
                    if p < 0.001:
                        p_str = "< 0.001"
                    else:
                        p_str = "{:.3f}".format(p)
                    print("$X_{:d}$ & $X_{:d}$ & ${:s}$ & {:s}{:s}".format(variable_to_fit, variable_to_test, p_str,
                                                                           str(p < alpha), r"\\"))
        print("\hline")
        print("\end{tabular}")