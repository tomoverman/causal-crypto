import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
def AR_fit(train_data, test_data, variable_to_fit, p):
    hist = train_data
    hist_test = test_data
    # get lagged values
    hist_lag = np.roll(hist, 1, 1)
    hist_lag_test = np.roll(hist_test, 1, 1)

    # get total number of variables
    dimension = train_data.shape[0]

    data_in = hist_lag[:, p:].T
    data_in_test = hist_lag_test[:, p:].T
    for i in range(p - 1):
        hist_lag2 = np.roll(hist_lag, i + 1, 1)
        data_in = np.concatenate((data_in, hist_lag2[:, p:].T), axis=1)

        hist_lag2_test = np.roll(hist_lag_test, i + 1, 1)
        data_in_test = np.concatenate((data_in_test, hist_lag2_test[:, p:].T), axis=1)

    # fit full model
    full_reg = LinearRegression().fit(data_in, hist[variable_to_fit, p:])

    # get predictions on data
    train_prediction = full_reg.predict(data_in)
    test_prediction = full_reg.predict(data_in_test)

    # get residuals
    train_residuals = train_prediction - hist[variable_to_fit, p:]
    test_residuals = test_prediction - hist_test[variable_to_fit, p:]

    # return residuals
    return train_prediction, train_residuals, test_prediction, test_residuals