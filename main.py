from Historic_Crypto import HistoricalData
from Historic_Crypto import Cryptocurrencies
from granger import granger
from sindy import sindy
from dmd import *
from ccm import *
from AR_model import *
import numpy as np
import matplotlib.pyplot as plt
from covid19dh import covid19
import csv
import sys

if sys.argv[1] == "granger_between":
    data = np.loadtxt('crypto_between.txt')
    t=np.arange(0,data.shape[1])
    lags = np.array([5,30,60,90])
    granger_model = granger(data)
    for l in lags:
        print("With time lag = " + str(l))
        granger_model.causality(l)

elif sys.argv[1]== "granger_covid_gt":
    data1 = np.loadtxt('crypto_covid.txt')
    data2 = np.loadtxt('crypto_trends.txt')
    data=np.zeros((6,data1.shape[1]))
    data[0:3,:]=data1
    data[3:6,:]=data2[2:,:]
    granger_model = granger(data)
    granger_model.causality(60)

elif sys.argv[1]=="ccm_between":
    Ls=np.linspace(50, 5050, 100).astype(int)
    corrs_eth=np.zeros(Ls.shape[0])
    corrs_btc = np.zeros(Ls.shape[0])
    for j,L in enumerate(Ls):
        data = np.loadtxt('crypto_between.txt')
        data=data[:,0:L]
        X=data[0,:]
        Y=data[1,:]
        E=3
        tau=1
        ccm = CCM(X, Y, E, tau)
        Y_pred = []
        for i in range(len(X)):
            y_pred = ccm.predict(i)
            Y_pred.append(y_pred)
        Y_pred = np.array(Y_pred)

        r, p = cor(Y[2:],Y_pred[2:])
        corrs_eth[j]=r
    for j,L in enumerate(Ls):
        data = np.loadtxt('crypto_between.txt')
        data=data[:,0:L]
        X=data[1,:]
        Y=data[0,:]
        E=3
        tau=1
        ccm = CCM(X, Y, E, tau)
        Y_pred = []
        for i in range(len(X)):
            y_pred = ccm.predict(i)
            Y_pred.append(y_pred)
        Y_pred = np.array(Y_pred)

        r, p = cor(Y[2:],Y_pred[2:])
        corrs_btc[j]=r
    plt.plot(Ls,corrs_eth)
    plt.plot(Ls, corrs_btc)
    plt.legend(["$ETH^*$|$M_{BTC}$","$BTC^*$|$M_{ETH}$"])
    plt.xlabel("L")
    plt.ylabel("Correlation Coefficient")
    plt.show()

elif sys.argv[1]=="ccm_covid_gt_btc":
    data1 = np.loadtxt('crypto_covid.txt')
    data2 = np.loadtxt('crypto_trends.txt')
    data = np.zeros((6, data1.shape[1]))
    data[0:3, :] = data1
    data[3:6, :] = data2[2:, :]

    Ls=np.linspace(50, 5050, 250).astype(int)
    corrs_btc_eth=np.zeros(Ls.shape[0])
    corrs_btc_covid = np.zeros(Ls.shape[0])
    corrs_btc_bitcoin = np.zeros(Ls.shape[0])
    corrs_btc_ethereum = np.zeros(Ls.shape[0])
    corrs_btc_elonmusk = np.zeros(Ls.shape[0])

    for j,L in enumerate(Ls):
        data1=data[:,0:L]
        X=data1[1,:]
        Y=data1[0,:]
        E=3
        tau=1
        ccm = CCM(X, Y, E, tau)
        Y_pred = []
        for i in range(len(X)):
            y_pred = ccm.predict(i)
            Y_pred.append(y_pred)
        Y_pred = np.array(Y_pred)
        r, p = cor(Y[2:],Y_pred[2:])
        corrs_btc_eth[j]=r

        X = data1[2, :]
        Y = data1[0, :]
        E = 3
        tau = 1
        ccm = CCM(X, Y, E, tau)
        Y_pred = []
        for i in range(len(X)):
            y_pred = ccm.predict(i)
            Y_pred.append(y_pred)
        Y_pred = np.array(Y_pred)
        r, p = cor(Y[2:], Y_pred[2:])
        corrs_btc_covid[j] = r

        X = data1[3, :]
        Y = data1[0, :]
        E = 3
        tau = 1
        ccm = CCM(X, Y, E, tau)
        Y_pred = []
        for i in range(len(X)):
            y_pred = ccm.predict(i)
            Y_pred.append(y_pred)
        Y_pred = np.array(Y_pred)
        r, p = cor(Y[2:], Y_pred[2:])
        corrs_btc_bitcoin[j] = r

        X = data1[4, :]
        Y = data1[0, :]
        E = 3
        tau = 1
        ccm = CCM(X, Y, E, tau)
        Y_pred = []
        for i in range(len(X)):
            y_pred = ccm.predict(i)
            Y_pred.append(y_pred)
        Y_pred = np.array(Y_pred)
        r, p = cor(Y[2:], Y_pred[2:])
        corrs_btc_ethereum[j] = r

        X = data1[5, :]
        Y = data1[0, :]
        E = 3
        tau = 1
        ccm = CCM(X, Y, E, tau)
        Y_pred = []
        for i in range(len(X)):
            y_pred = ccm.predict(i)
            Y_pred.append(y_pred)
        Y_pred = np.array(Y_pred)
        r, p = cor(Y[2:], Y_pred[2:])
        corrs_btc_elonmusk[j] = r

    plt.plot(Ls,corrs_btc_eth)
    plt.plot(Ls, corrs_btc_covid)
    plt.plot(Ls, corrs_btc_bitcoin)
    plt.plot(Ls, corrs_btc_ethereum)
    plt.plot(Ls, corrs_btc_elonmusk)
    plt.legend(["$BTC^*$|$M_{ETH}$","$BTC^*$|$M_{COVID}$","$BTC^*$|$M_{gt-bitcoin}$", "$BTC^*$|$M_{gt-ethereum}$","$BTC^*$|$M_{gt-elonmusk}$"])
    plt.xlabel("L")
    plt.ylabel("Correlation Coefficient")
    plt.show()

elif sys.argv[1]=="ccm_covid_gt_eth":
    data1 = np.loadtxt('crypto_covid.txt')
    data2 = np.loadtxt('crypto_trends.txt')
    data = np.zeros((6, data1.shape[1]))
    data[0:3, :] = data1
    data[3:6, :] = data2[2:, :]

    Ls=np.linspace(50, 5050, 250).astype(int)
    corrs_eth_btc=np.zeros(Ls.shape[0])
    corrs_eth_covid = np.zeros(Ls.shape[0])
    corrs_eth_bitcoin = np.zeros(Ls.shape[0])
    corrs_eth_ethereum = np.zeros(Ls.shape[0])
    corrs_eth_elonmusk = np.zeros(Ls.shape[0])

    for j,L in enumerate(Ls):
        data1=data[:,0:L]
        X=data1[0,:]
        Y=data1[1,:]
        E=3
        tau=1
        ccm = CCM(X, Y, E, tau)
        Y_pred = []
        for i in range(len(X)):
            y_pred = ccm.predict(i)
            Y_pred.append(y_pred)
        Y_pred = np.array(Y_pred)
        r, p = cor(Y[2:],Y_pred[2:])
        corrs_eth_btc[j]=r

        X = data1[2, :]
        Y = data1[1, :]
        E = 3
        tau = 1
        ccm = CCM(X, Y, E, tau)
        Y_pred = []
        for i in range(len(X)):
            y_pred = ccm.predict(i)
            Y_pred.append(y_pred)
        Y_pred = np.array(Y_pred)
        r, p = cor(Y[2:], Y_pred[2:])
        corrs_eth_covid[j] = r

        X = data1[3, :]
        Y = data1[1, :]
        E = 3
        tau = 1
        ccm = CCM(X, Y, E, tau)
        Y_pred = []
        for i in range(len(X)):
            y_pred = ccm.predict(i)
            Y_pred.append(y_pred)
        Y_pred = np.array(Y_pred)
        r, p = cor(Y[2:], Y_pred[2:])
        corrs_eth_bitcoin[j] = r

        X = data1[4, :]
        Y = data1[1, :]
        E = 3
        tau = 1
        ccm = CCM(X, Y, E, tau)
        Y_pred = []
        for i in range(len(X)):
            y_pred = ccm.predict(i)
            Y_pred.append(y_pred)
        Y_pred = np.array(Y_pred)
        r, p = cor(Y[2:], Y_pred[2:])
        corrs_eth_ethereum[j] = r

        X = data1[5, :]
        Y = data1[1, :]
        E = 3
        tau = 1
        ccm = CCM(X, Y, E, tau)
        Y_pred = []
        for i in range(len(X)):
            y_pred = ccm.predict(i)
            Y_pred.append(y_pred)
        Y_pred = np.array(Y_pred)
        r, p = cor(Y[2:], Y_pred[2:])
        corrs_eth_elonmusk[j] = r

    plt.plot(Ls,corrs_eth_btc)
    plt.plot(Ls, corrs_eth_covid)
    plt.plot(Ls, corrs_eth_bitcoin)
    plt.plot(Ls, corrs_eth_ethereum)
    plt.plot(Ls, corrs_eth_elonmusk)
    plt.legend(["$ETH^*$|$M_{BTC}$","$ETH^*$|$M_{COVID}$","$ETH^*$|$M_{gt-bitcoin}$", "$ETH^*$|$M_{gt-ethereum}$","$ETH^*$|$M_{gt-elonmusk}$"])
    plt.xlabel("L")
    plt.ylabel("Correlation Coefficient")
    plt.show()

if sys.argv[1] == "granger_no_data":
    data = np.loadtxt('crypto_between.txt')
    t=np.arange(0,data.shape[1])
    granger_model = granger(data[:,:100])
    print("With data legnth = 100")
    granger_model.causality(60)

elif sys.argv[1]=="sindy":
    data1 = np.loadtxt('crypto_covid.txt')
    data2 = np.loadtxt('crypto_trends.txt')
    data = np.zeros((3, data1.shape[1]))
    data[0:2, :] = data1[0:2,:]
    data[2, :] = data2[2, :]
    t = np.arange(0, 600)
    pred,test_pred = sindy(data[:,:600],t)
    plt.figure(1)
    plt.plot(pred.T[0,:])
    plt.plot(data[0,:600])
    plt.xlabel("Day")
    plt.ylabel("BTC Volume")
    plt.legend(["Model Prediction", "True Values"])

    plt.figure(2)
    plt.plot(test_pred.T[0, :])
    plt.plot(data[0, 600:])
    plt.xlabel("Day")
    plt.ylabel("BTC Volume")
    plt.legend(["Model Prediction", "True Values"])

    plt.show()

elif sys.argv[1]=="dmd":
    #use 650 data points to train and then test on 800 points
    data1 = np.loadtxt('crypto_covid.txt')
    data2 = np.loadtxt('crypto_trends.txt')
    data = np.zeros((3, data1.shape[1]))
    data[0:2, :] = data1[0:2, :]
    data[2, :] = data2[2, :]
    t = np.arange(0, data.shape[1])

    pred_step = 200
    r = 3
    mat_hat = DMD_pred(data[:,:600], r, pred_step)
    print(mat_hat.shape)
    plt.figure(1)
    plt.plot(mat_hat[0, :600])
    plt.plot(data[0, :600])
    plt.xlabel("Day")
    plt.ylabel("BTC Volume")
    plt.legend(["Model Prediction","True Values"])

    plt.figure(2)
    plt.plot(mat_hat[0,600:])
    plt.plot(data[0, 600:])
    plt.xlabel("Day")
    plt.ylabel("BTC Volume")
    plt.legend(["Model Prediction", "True Values"])
    plt.show()



elif sys.argv[1]=="ar":
    data1 = np.loadtxt('crypto_covid.txt')
    data2 = np.loadtxt('crypto_trends.txt')
    data = np.zeros((3, data1.shape[1]))
    data[0:2, :] = data1[0:2, :]
    data[2, :] = data2[2, :]
    t = np.arange(0, data.shape[1])
    train_data = data[:, :600]
    test_data = data[:, 600:]
    lags_list=np.array([2,5,10,20,30,40,50,60,70,80,90,100])
    test_residuals=[]
    for lags in lags_list:
        train_pred,train_res,test_pred,test_res = AR_fit(train_data,test_data,0,lags)
        test_residuals.append(np.sum(test_res**2)/test_res.shape[0])


    plt.figure(1)
    plt.semilogy(lags_list,test_residuals)
    plt.xlabel("Number of Lags")
    plt.ylabel("Testing SSR")

    best_lag=30
    train_pred, train_res, test_pred, test_res = AR_fit(train_data, test_data, 0, best_lag)
    plt.figure(2)
    plt.plot(test_pred)
    plt.plot(test_data[0,best_lag:])
    plt.legend(["Model Prediction", "True Data"])
    plt.xlabel("Day")
    plt.ylabel("BTC Trading Volume")
    plt.show()




# eth = HistoricalData('ETH-USD',86400,'2020-01-03-00-00').retrieve_data()
    # eth_vol = np.array(eth['volume'])
    # btc = HistoricalData('BTC-USD',86400,'2020-01-03-00-00').retrieve_data()
    # btc_vol = np.array(btc['volume'])
    # data = np.zeros((5, eth_vol.shape[0]))
    # data[0, :] = eth_vol
    # data[1, :] = btc_vol
    #
    # btc_path = "GT_bitcoin.csv"
    # eth_path = "GT_ethereum.csv"
    # em_path = "GT_elonmusk.csv"
    #
    # btc_data=[]
    # eth_data=[]
    # em_data=[]
    # with open(btc_path) as csvfile:
    #     reader = csv.reader(csvfile)
    #     for i, row in enumerate(reader):
    #         if i == 2:
    #             print(', '.join(row))
    #         elif i==3:
    #             for j in range(0,9):
    #                 btc_data.append(row[1])
    #         elif i>=4:
    #             for j in range(0,7):
    #                 btc_data.append(row[1])
    # with open(eth_path) as csvfile:
    #     reader = csv.reader(csvfile)
    #     for i, row in enumerate(reader):
    #         if i == 2:
    #             print(', '.join(row))
    #         elif i==3:
    #             for j in range(0,9):
    #                 eth_data.append(row[1])
    #         elif i>=4:
    #             for j in range(0,7):
    #                 eth_data.append(row[1])
    # with open(em_path) as csvfile:
    #     reader = csv.reader(csvfile)
    #     for i, row in enumerate(reader):
    #         if i == 2:
    #             print(', '.join(row))
    #         elif i==3:
    #             for j in range(0,9):
    #                 em_data.append(row[1])
    #         elif i>=4:
    #             for j in range(0,7):
    #                 em_data.append(row[1])
    #
    # btc=np.array(btc_data)
    # eth=np.array(eth_data)
    # em=np.array(em_data)
    # data[2, :] = btc_data
    # data[3,:] = eth_data
    # data[4,:] = em_data
    # np.savetxt('crypto_trends.txt', data, fmt='%d')