import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt

def sindy(X,t):

    fourier_library = ps.FourierLibrary(n_frequencies=1)
    poly_library = ps.PolynomialLibrary(include_interaction=True,degree=2)

    combined_library = poly_library

    stlsq_optimizer = ps.STLSQ(threshold=.0001)
    model = ps.SINDy(feature_library=combined_library, feature_names=["btc", "eth", "gt-btc"],optimizer=stlsq_optimizer)
    model.fit(np.transpose(X), t=t)
    model.print()

    #return the model evaluated at all 800 time points
    train_pred= model.simulate(X[:,0], t,integrator_kws={'atol': 1e-3, 'method': 'LSODA', 'rtol': 1e-3})
    t=np.arange(600,800)
    test_pred=model.simulate(X[:,-1], t,integrator_kws={'atol': 1e-10, 'method': 'LSODA', 'rtol': 1e-10})
    return train_pred, test_pred